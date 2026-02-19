using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using MobileNetV3.Core.Configuration;

namespace MobileNetV3.Core.Export;

/// <summary>
/// Утилита для верификации и анализа экспортированной ONNX-модели.
/// Позволяет убедиться, что финальная модель корректно принимает входы
/// и возвращает ожидаемые выходы нужных размерностей.
/// </summary>
public sealed class ModelExporter
{
    private readonly TrainingConfig _config;
    private readonly ILogger<ModelExporter> _logger;

    public ModelExporter(TrainingConfig config, ILogger<ModelExporter> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Загружает экспортированную модель и запускает один тестовый прогон
    /// для верификации корректности структуры.
    /// </summary>
    public ModelInfo Verify(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Модель не найдена: {modelPath}");

        using var session = new InferenceSession(modelPath);

        var inputs  = session.InputMetadata;
        var outputs = session.OutputMetadata;

        _logger.LogInformation("Верификация модели: {Path}", modelPath);
        _logger.LogInformation("Входы модели:");
        foreach (var (name, meta) in inputs)
            _logger.LogInformation("  {Name}: {Type} {Shape}",
                name, meta.ElementType, string.Join("x", meta.Dimensions));

        _logger.LogInformation("Выходы модели:");
        foreach (var (name, meta) in outputs)
            _logger.LogInformation("  {Name}: {Type} {Shape}",
                name, meta.ElementType, string.Join("x", meta.Dimensions));

        // Тестовый прогон с нулевым тензором [1, 3, 224, 224]
        var dummyData  = new float[1 * 3 * _config.ImageSize * _config.ImageSize];
        var dummyShape = new long[] { 1, 3, _config.ImageSize, _config.ImageSize };

        using var dummyTensor = OrtValue.CreateTensorValueFromMemory(dummyData, dummyShape);

        var inputDict = new Dictionary<string, OrtValue>
            { [_config.InputNodeName] = dummyTensor };

        using var results = session.Run(new RunOptions(), inputDict, session.OutputNames);

        var outputSpan = results[0].GetTensorDataAsSpan<float>();

        _logger.LogInformation(
            "Тестовый прогон успешен. Размер выхода: {Size} (ожидается {Expected})",
            outputSpan.Length, _config.NumClasses);

        return new ModelInfo
        {
            ModelPath   = modelPath,
            InputNames  = inputs.Keys.ToList(),
            OutputNames = outputs.Keys.ToList(),
            NumClasses  = outputSpan.Length,
            FileSizeKb  = new FileInfo(modelPath).Length / 1024
        };
    }
}

/// <summary>Метаданные проверенной ONNX-модели</summary>
public sealed class ModelInfo
{
    public string ModelPath { get; init; } = string.Empty;
    public IReadOnlyList<string> InputNames { get; init; } = [];
    public IReadOnlyList<string> OutputNames { get; init; } = [];
    public int NumClasses { get; init; }
    public long FileSizeKb { get; init; }
}
