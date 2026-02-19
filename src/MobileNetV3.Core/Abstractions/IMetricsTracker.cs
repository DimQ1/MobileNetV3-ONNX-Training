namespace MobileNetV3.Core.Abstractions;

/// <summary>
/// Отслеживает накопленные метрики за эпоху (loss, accuracy)
/// </summary>
public interface IMetricsTracker
{
    void Reset();
    void Update(float loss, int correctPredictions, int batchSize);
    float AverageLoss { get; }
    float Accuracy { get; }
    int TotalSamples { get; }
}
