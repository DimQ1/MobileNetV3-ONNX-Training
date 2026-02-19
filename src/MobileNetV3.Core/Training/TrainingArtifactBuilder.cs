using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Configuration;

namespace MobileNetV3.Core.Training;

/// <summary>
/// Генерирует артефакты обучения ONNX Runtime Training:
/// training_model.onnx, eval_model.onnx, optimizer_model.onnx, checkpoint/.
///
/// Примечание: ONNX Runtime Training требует предварительной генерации
/// этих графов через Python-утилиту orttraining.artifacts.generate_artifacts().
/// Данный класс обеспечивает валидацию наличия артефактов и предоставляет
/// инструкции по их генерации в случае отсутствия.
/// </summary>
public sealed class TrainingArtifactBuilder
{
    private readonly TrainingConfig _config;
    private readonly ILogger<TrainingArtifactBuilder> _logger;

    public TrainingArtifactBuilder(
        TrainingConfig config,
        ILogger<TrainingArtifactBuilder> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Проверяет наличие всех необходимых артефактов обучения.
    /// Бросает <see cref="FileNotFoundException"/> с инструкцией, если артефакты отсутствуют.
    /// </summary>
    public void EnsureArtifactsExist()
    {
        var requiredFiles = new[]
        {
            (_config.BaseModelPath,      "Базовая ONNX-модель"),
            (_config.TrainingModelPath,  "Граф прямого прохода (training)"),
            (_config.EvalModelPath,      "Граф оценки (eval)"),
            (_config.OptimizerModelPath, "Граф оптимизатора"),
        };

        var missing = requiredFiles
            .Where(f => !File.Exists(f.Item1))
            .ToList();

        if (!Directory.Exists(_config.CheckpointDir) ||
            !Directory.EnumerateFiles(_config.CheckpointDir).Any())
        {
            missing.Add((_config.CheckpointDir, "Директория checkpoint"));
        }

        if (missing.Count == 0)
        {
            _logger.LogInformation("Все артефакты обучения найдены.");
            return;
        }

        var missingList = string.Join("\n  ", missing.Select(m => $"[{m.Item2}]: {m.Item1}"));

        throw new FileNotFoundException(
            $"""
            Отсутствуют артефакты обучения ONNX Runtime Training:
              {missingList}

            Для генерации артефактов выполните Python-скрипт:
              python scripts/generate_training_artifacts.py

            Подробнее: https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html
            """);
    }

    /// <summary>
    /// Создаёт необходимые директории для артефактов и вывода.
    /// </summary>
    public void EnsureDirectoriesExist()
    {
        var dirs = new[]
        {
            Path.GetDirectoryName(_config.TrainingModelPath),
            Path.GetDirectoryName(_config.EvalModelPath),
            Path.GetDirectoryName(_config.OptimizerModelPath),
            _config.CheckpointDir,
            Path.GetDirectoryName(_config.OutputModelPath),
        };

        foreach (var dir in dirs.Where(d => !string.IsNullOrWhiteSpace(d)))
        {
            Directory.CreateDirectory(dir!);
            _logger.LogDebug("Директория создана/проверена: {Dir}", dir);
        }
    }
}
