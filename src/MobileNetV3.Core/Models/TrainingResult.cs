namespace MobileNetV3.Core.Models;

/// <summary>Результаты одной эпохи обучения</summary>
public sealed class EpochResult
{
    public int Epoch { get; init; }
    public float TrainLoss { get; init; }
    public float TrainAccuracy { get; init; }
    public float ValidationLoss { get; init; }
    public float ValidationAccuracy { get; init; }
    public TimeSpan Duration { get; init; }
    public float LearningRate { get; init; }

    public override string ToString() =>
        $"Epoch {Epoch:D3} | " +
        $"Train Loss: {TrainLoss:F4} | Train Acc: {TrainAccuracy:P2} | " +
        $"Val Loss:   {ValidationLoss:F4} | Val Acc:   {ValidationAccuracy:P2} | " +
        $"LR: {LearningRate:E2} | Time: {Duration:mm\\:ss}";
}

/// <summary>Итоговая сводка по всему обучению</summary>
public sealed class TrainingReport
{
    public string ModelOutputPath { get; init; } = string.Empty;
    public IReadOnlyList<EpochResult> EpochHistory { get; init; } = [];
    public float BestValidationAccuracy { get; init; }
    public int BestEpoch { get; init; }
    public TimeSpan TotalDuration { get; init; }
    public IReadOnlyDictionary<string, float> PerClassAccuracy { get; init; }
        = new Dictionary<string, float>();
}
