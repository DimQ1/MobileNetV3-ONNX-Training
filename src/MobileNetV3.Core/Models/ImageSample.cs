namespace MobileNetV3.Core.Models;

/// <summary>
/// Один обработанный образец датасета (тензор + метка)
/// </summary>
public sealed class ImageSample
{
    /// <summary>Нормализованные пиксели в формате CHW float32 [C, H, W]</summary>
    public required float[] Tensor { get; init; }

    /// <summary>Целочисленная метка класса [0..NumClasses-1]</summary>
    public required long Label { get; init; }

    /// <summary>Исходный путь к файлу (для диагностики)</summary>
    public required string FilePath { get; init; }

    /// <summary>Имя класса (для удобства логирования)</summary>
    public required string ClassName { get; init; }
}

/// <summary>
/// Батч из нескольких образцов, готовый для передачи в TrainingSession
/// </summary>
public sealed class TrainingBatch
{
    /// <summary>
    /// Тензор изображений [BatchSize, Channels, Height, Width] в формате NCHW
    /// </summary>
    public required float[] Images { get; init; }

    /// <summary>Тензор меток [BatchSize] (int64)</summary>
    public required long[] Labels { get; init; }

    /// <summary>Фактический размер батча</summary>
    public required int BatchSize { get; init; }
}
