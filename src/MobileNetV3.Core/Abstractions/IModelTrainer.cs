using MobileNetV3.Core.Models;

namespace MobileNetV3.Core.Abstractions;

/// <summary>
/// Контракт тренера модели. Инкапсулирует полный цикл fine-tuning.
/// </summary>
public interface IModelTrainer : IDisposable
{
    /// <summary>
    /// Запускает полный цикл дообучения.
    /// </summary>
    Task<TrainingReport> TrainAsync(
        IReadOnlyList<ImageSample> trainSamples,
        IReadOnlyList<ImageSample> valSamples,
        IProgress<EpochResult>? progress = null,
        CancellationToken ct = default);

    /// <summary>
    /// Экспортирует обученную модель в ONNX для инференса.
    /// </summary>
    void ExportModel(string outputPath);
}
