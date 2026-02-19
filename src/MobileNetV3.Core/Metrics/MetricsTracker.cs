using MobileNetV3.Core.Abstractions;

namespace MobileNetV3.Core.Metrics;

/// <summary>
/// Потокобезопасный трекер накопленных метрик за эпоху.
/// </summary>
public sealed class MetricsTracker : IMetricsTracker
{
    private readonly object _lock = new();

    private double _totalLoss;
    private int _totalCorrect;
    private int _totalSamples;

    public void Reset()
    {
        lock (_lock)
        {
            _totalLoss    = 0;
            _totalCorrect = 0;
            _totalSamples = 0;
        }
    }

    public void Update(float loss, int correctPredictions, int batchSize)
    {
        lock (_lock)
        {
            _totalLoss    += loss * batchSize; // взвешенная сумма для усреднения
            _totalCorrect += correctPredictions;
            _totalSamples += batchSize;
        }
    }

    public float AverageLoss => _totalSamples == 0
        ? 0f
        : (float)(_totalLoss / _totalSamples);

    public float Accuracy => _totalSamples == 0
        ? 0f
        : (float)_totalCorrect / _totalSamples;

    public int TotalSamples => _totalSamples;
}
