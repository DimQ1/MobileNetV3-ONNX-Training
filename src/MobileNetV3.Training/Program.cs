using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Export;
using MobileNetV3.Core.Metrics;
using MobileNetV3.Core.Models;
using MobileNetV3.Core.Preprocessing;
using MobileNetV3.Core.Training;

// ─── Загрузка конфигурации ────────────────────────────────────────────────────

const string ConfigPath = "training_config.json";
TrainingConfig config;

if (File.Exists(ConfigPath))
{
    var json = await File.ReadAllTextAsync(ConfigPath);
    config = JsonSerializer.Deserialize<TrainingConfig>(json,
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
        ?? new TrainingConfig();
    Console.WriteLine($"Конфигурация загружена из: {ConfigPath}");
}
else
{
    config = new TrainingConfig();
    // Сохраняем конфиг по умолчанию для ручной правки
    var json = JsonSerializer.Serialize(config,
        new JsonSerializerOptions { WriteIndented = true });
    await File.WriteAllTextAsync(ConfigPath, json);
    Console.WriteLine($"Создан конфиг по умолчанию: {ConfigPath}. Отредактируйте и перезапустите.");
}

// ─── Настройка DI-контейнера ──────────────────────────────────────────────────

var services = new ServiceCollection();

services.AddLogging(builder =>
{
    builder.AddSimpleConsole(opt =>
    {
        opt.TimestampFormat = "[HH:mm:ss] ";
        opt.SingleLine = false;
        opt.IncludeScopes = false;
    });
    builder.SetMinimumLevel(LogLevel.Information);
});

services.AddSingleton(config);
services.AddSingleton<IImagePreprocessor, ImagePreprocessor>();
services.AddSingleton<IDatasetLoader, DatasetLoader>();
services.AddSingleton<IMetricsTracker, MetricsTracker>();
services.AddTransient<LearningRateScheduler>();
services.AddTransient<TrainingArtifactBuilder>();
services.AddTransient<ModelExporter>();

// Тренер создаётся с двумя отдельными MetricsTracker (train / val)
services.AddTransient<IModelTrainer>(sp =>
{
    return new ModelTrainer(
        sp.GetRequiredService<TrainingConfig>(),
        sp.GetRequiredService<IDatasetLoader>(),
        new MetricsTracker(),   // train metrics
        new MetricsTracker(),   // val metrics
        sp.GetRequiredService<LearningRateScheduler>(),
        sp.GetRequiredService<ILogger<ModelTrainer>>());
});

await using var serviceProvider = services.BuildServiceProvider();
var logger = serviceProvider.GetRequiredService<ILogger<Program>>();

// ─── Проверка артефактов ──────────────────────────────────────────────────────

var artifactBuilder = serviceProvider.GetRequiredService<TrainingArtifactBuilder>();

try
{
    artifactBuilder.EnsureDirectoriesExist();
    artifactBuilder.EnsureArtifactsExist();
}
catch (FileNotFoundException ex)
{
    logger.LogError("{Message}", ex.Message);
    Console.WriteLine("\nНажмите любую клавишу для выхода...");
    Console.ReadKey();
    return;
}

// ─── Загрузка датасета ────────────────────────────────────────────────────────

logger.LogInformation("═══ Загрузка датасета ═══");

var datasetLoader = serviceProvider.GetRequiredService<IDatasetLoader>();

IReadOnlyList<ImageSample> trainSamples;
IReadOnlyList<ImageSample> valSamples;

using var loadCts = new CancellationTokenSource(TimeSpan.FromMinutes(10));

try
{
    (trainSamples, valSamples) = await datasetLoader.LoadAsync(
        config.DatasetRootDir,
        config.ValidationSplit,
        loadCts.Token);
}
catch (Exception ex)
{
    logger.LogError("Ошибка загрузки датасета: {Message}", ex.Message);
    return;
}

if (trainSamples.Count == 0)
{
    logger.LogError("Датасет пуст. Проверьте директорию: {Dir}", config.DatasetRootDir);
    return;
}

logger.LogInformation("Train: {T} образцов | Validation: {V} образцов",
    trainSamples.Count, valSamples.Count);

// ─── Запуск дообучения ────────────────────────────────────────────────────────

logger.LogInformation("═══ Запуск дообучения ═══");

using var trainer = serviceProvider.GetRequiredService<IModelTrainer>();
using var trainCts = new CancellationTokenSource();

// Обработка Ctrl+C: плавная остановка
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    logger.LogWarning("Получен сигнал остановки. Завершаем текущую эпоху...");
    trainCts.Cancel();
};

// Прогресс в консоли
var progressHandler = new Progress<EpochResult>(result =>
{
    Console.ForegroundColor = result.ValidationAccuracy > 0.9f
        ? ConsoleColor.Green
        : ConsoleColor.White;
    Console.WriteLine($"  ► {result}");
    Console.ResetColor();
});

TrainingReport report;

try
{
    report = await trainer.TrainAsync(trainSamples, valSamples, progressHandler, trainCts.Token);
}
catch (OperationCanceledException)
{
    logger.LogWarning("Обучение остановлено пользователем.");
    return;
}
catch (Exception ex)
{
    logger.LogError(ex, "Критическая ошибка во время обучения.");
    return;
}

// ─── Верификация финальной модели ─────────────────────────────────────────────

logger.LogInformation("═══ Верификация экспортированной модели ═══");

var exporter = serviceProvider.GetRequiredService<ModelExporter>();

try
{
    var modelInfo = exporter.Verify(report.ModelOutputPath);
    logger.LogInformation(
        "Модель верифицирована: {Classes} классов, размер: {Size} KB",
        modelInfo.NumClasses, modelInfo.FileSizeKb);
}
catch (Exception ex)
{
    logger.LogError("Ошибка верификации: {Message}", ex.Message);
}

// ─── Итоговый отчёт ───────────────────────────────────────────────────────────

logger.LogInformation("═══ Итоговый отчёт ═══");
logger.LogInformation("Лучшая точность: {Acc:P2} (epoch {E})",
    report.BestValidationAccuracy, report.BestEpoch);
logger.LogInformation("Точность по классам:");

foreach (var (cls, acc) in report.PerClassAccuracy)
    logger.LogInformation("  {Class,-15} → {Acc:P2}", cls, acc);

logger.LogInformation("Модель готова к деплою: {Path}", report.ModelOutputPath);
