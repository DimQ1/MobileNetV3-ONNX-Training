using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Models;

namespace MobileNetV3.Core.Data;

/// <summary>
/// Загружает датасет изображений из папок (каждая папка = класс).
/// Выполняет стратифицированное разделение train/validation.
/// </summary>
public sealed class DatasetLoader : IDatasetLoader
{
    private static readonly string[] SupportedExtensions =
        [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"];

    private readonly TrainingConfig _config;
    private readonly IImagePreprocessor _preprocessor;
    private readonly ILogger<DatasetLoader> _logger;
    private readonly Random _rng = new(42);

    public DatasetLoader(
        TrainingConfig config,
        IImagePreprocessor preprocessor,
        ILogger<DatasetLoader> logger)
    {
        _config = config;
        _preprocessor = preprocessor;
        _logger = logger;
    }

    /// <inheritdoc/>
    public async Task<(IReadOnlyList<ImageSample> Train, IReadOnlyList<ImageSample> Validation)>
        LoadAsync(string rootDir, float validationSplit, CancellationToken ct = default)
    {
        if (!Directory.Exists(rootDir))
            throw new DirectoryNotFoundException($"Директория датасета не найдена: {rootDir}");

        _logger.LogInformation("Загрузка датасета из: {Dir}", rootDir);

        // Собираем файлы по классам
        var classDirs = Directory.GetDirectories(rootDir)
            .OrderBy(d => d)
            .ToArray();

        if (classDirs.Length == 0)
            throw new InvalidOperationException(
                $"В директории '{rootDir}' не найдено папок с классами.");

        _logger.LogInformation("Найдено классов: {Count}", classDirs.Length);

        // Параллельная обработка файлов по классам
        var allSamples = new List<ImageSample>();
        var semaphore = new SemaphoreSlim(Environment.ProcessorCount, Environment.ProcessorCount);

        foreach (var classDir in classDirs)
        {
            ct.ThrowIfCancellationRequested();

            string className = Path.GetFileName(classDir);

            // Ищем индекс класса в конфиге (если класс не в конфиге — пропускаем)
            int classIndex = Array.IndexOf(_config.ClassLabels, className);
            if (classIndex < 0)
            {
                _logger.LogWarning(
                    "Папка '{Name}' не соответствует ни одному классу из конфига. Пропускаем.",
                    className);
                continue;
            }

            var files = GetImageFiles(classDir);
            _logger.LogInformation("  Класс '{Class}' (idx={Idx}): {Count} изображений",
                className, classIndex, files.Count);

            // Параллельный препроцессинг с ограничением потоков
            var tasks = files.Select(async filePath =>
            {
                await semaphore.WaitAsync(ct);
                try
                {
                    return await Task.Run(() =>
                    {
                        try
                        {
                            var tensor = _preprocessor.Preprocess(filePath, augment: false);
                            return new ImageSample
                            {
                                Tensor    = tensor,
                                Label     = classIndex,
                                FilePath  = filePath,
                                ClassName = className
                            };
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning("Не удалось обработать {File}: {Err}",
                                filePath, ex.Message);
                            return null;
                        }
                    }, ct);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var results = await Task.WhenAll(tasks);
            allSamples.AddRange(results.Where(s => s is not null)!);
        }

        _logger.LogInformation("Всего загружено образцов: {Total}", allSamples.Count);

        // Стратифицированное разделение: пропорции классов сохраняются
        var (train, val) = StratifiedSplit(allSamples, validationSplit);

        _logger.LogInformation(
            "Разделение: Train={Train}, Validation={Val}", train.Count, val.Count);

        return (train, val);
    }

    /// <inheritdoc/>
    public IEnumerable<TrainingBatch> GetBatches(
        IReadOnlyList<ImageSample> samples,
        int batchSize,
        bool shuffle = true)
    {
        var indices = Enumerable.Range(0, samples.Count).ToList();

        if (shuffle)
        {
            // Fisher-Yates shuffle
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = _rng.Next(0, i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int tensorSize = _config.ImageSize * _config.ImageSize * 3; // C * H * W

        for (int start = 0; start < indices.Count; start += batchSize)
        {
            int end = Math.Min(start + batchSize, indices.Count);
            int actualBatchSize = end - start;

            var images = new float[actualBatchSize * tensorSize];
            var labels = new long[actualBatchSize];

            for (int i = 0; i < actualBatchSize; i++)
            {
                var sample = samples[indices[start + i]];
                Array.Copy(sample.Tensor, 0, images, i * tensorSize, tensorSize);
                labels[i] = sample.Label;
            }

            yield return new TrainingBatch
            {
                Images    = images,
                Labels    = labels,
                BatchSize = actualBatchSize
            };
        }
    }

    // ─── Вспомогательные методы ──────────────────────────────────────────────

    private static List<string> GetImageFiles(string directory)
    {
        return Directory
            .EnumerateFiles(directory, "*.*", SearchOption.AllDirectories)
            .Where(f => SupportedExtensions.Contains(
                Path.GetExtension(f).ToLowerInvariant()))
            .ToList();
    }

    private (List<ImageSample> Train, List<ImageSample> Val)
        StratifiedSplit(List<ImageSample> samples, float valSplit)
    {
        var train = new List<ImageSample>();
        var val   = new List<ImageSample>();

        // Группируем по классам и делим каждый класс отдельно
        var byClass = samples.GroupBy(s => s.Label);

        foreach (var group in byClass)
        {
            var shuffled = group.OrderBy(_ => _rng.Next()).ToList();
            int valCount = Math.Max(1, (int)(shuffled.Count * valSplit));

            val.AddRange(shuffled.Take(valCount));
            train.AddRange(shuffled.Skip(valCount));
        }

        // Финальное перемешивание
        train = [.. train.OrderBy(_ => _rng.Next())];
        val   = [.. val.OrderBy(_ => _rng.Next())];

        return (train, val);
    }
}
