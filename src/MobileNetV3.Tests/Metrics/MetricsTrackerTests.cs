using MobileNetV3.Core.Metrics;

namespace MobileNetV3.Tests.Metrics;

public sealed class MetricsTrackerTests
{
    private readonly MetricsTracker _tracker = new();

    [Fact]
    public void InitialState_AllMetricsAreZero()
    {
        Assert.Equal(0f, _tracker.AverageLoss);
        Assert.Equal(0f, _tracker.Accuracy);
        Assert.Equal(0, _tracker.TotalSamples);
    }

    [Fact]
    public void Update_SingleBatch_ComputesCorrectLoss()
    {
        _tracker.Update(loss: 1.5f, correctPredictions: 8, batchSize: 10);

        Assert.Equal(1.5f, _tracker.AverageLoss, precision: 4);
    }

    [Fact]
    public void Update_SingleBatch_ComputesCorrectAccuracy()
    {
        _tracker.Update(loss: 1.0f, correctPredictions: 7, batchSize: 10);

        Assert.Equal(0.7f, _tracker.Accuracy, precision: 4);
    }

    [Fact]
    public void Update_MultipleBatches_WeightedAverageLoss()
    {
        // Батч 1: loss=2.0, size=10 → взвешенная сумма = 20
        // Батч 2: loss=1.0, size=10 → взвешенная сумма = 10
        // Среднее = 30 / 20 = 1.5
        _tracker.Update(loss: 2.0f, correctPredictions: 5, batchSize: 10);
        _tracker.Update(loss: 1.0f, correctPredictions: 8, batchSize: 10);

        Assert.Equal(1.5f, _tracker.AverageLoss, precision: 4);
    }

    [Fact]
    public void Update_MultipleBatches_AccuracyAccumulates()
    {
        _tracker.Update(loss: 1.0f, correctPredictions: 6, batchSize: 10);
        _tracker.Update(loss: 1.0f, correctPredictions: 8, batchSize: 10);

        // (6 + 8) / 20 = 0.7
        Assert.Equal(0.7f, _tracker.Accuracy, precision: 4);
    }

    [Fact]
    public void Update_UnEqualBatchSizes_WeightsCorrectly()
    {
        // Батч 1: 16 образцов, loss=2.0
        // Батч 2: 4 образца, loss=1.0
        // Среднее взвешенное: (2.0*16 + 1.0*4) / 20 = 36/20 = 1.8
        _tracker.Update(loss: 2.0f, correctPredictions: 10, batchSize: 16);
        _tracker.Update(loss: 1.0f, correctPredictions: 3, batchSize: 4);

        Assert.Equal(1.8f, _tracker.AverageLoss, precision: 4);
    }

    [Fact]
    public void Reset_ClearsAllMetrics()
    {
        _tracker.Update(loss: 1.0f, correctPredictions: 5, batchSize: 10);
        _tracker.Reset();

        Assert.Equal(0f, _tracker.AverageLoss);
        Assert.Equal(0f, _tracker.Accuracy);
        Assert.Equal(0, _tracker.TotalSamples);
    }

    [Fact]
    public void TotalSamples_AccumulatesAcrossBatches()
    {
        _tracker.Update(1.0f, 5, 10);
        _tracker.Update(1.0f, 5, 16);
        _tracker.Update(1.0f, 5, 7);

        Assert.Equal(33, _tracker.TotalSamples);
    }

    [Fact]
    public void AverageLoss_WhenNoSamples_ReturnsZero()
    {
        Assert.Equal(0f, _tracker.AverageLoss);
    }

    [Fact]
    public void Accuracy_WhenNoSamples_ReturnsZero()
    {
        Assert.Equal(0f, _tracker.Accuracy);
    }

    [Fact]
    public void Update_PerfectAccuracy_Returns1()
    {
        _tracker.Update(loss: 0.01f, correctPredictions: 32, batchSize: 32);

        Assert.Equal(1.0f, _tracker.Accuracy, precision: 4);
    }

    [Fact]
    public void Update_ZeroCorrect_ReturnsZeroAccuracy()
    {
        _tracker.Update(loss: 3.0f, correctPredictions: 0, batchSize: 32);

        Assert.Equal(0f, _tracker.Accuracy, precision: 4);
    }
}
