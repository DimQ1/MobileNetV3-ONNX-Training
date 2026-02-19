using MobileNetV3.Core.Models;

namespace MobileNetV3.Core.Abstractions;

/// <summary>
/// Контракт загрузчика датасета.
/// Ожидаемая структура папок:
/// <code>
/// dataset/
///   Clean/
///     img001.jpg
///     img002.png
///   DryDirt/
///     ...
///   WaterDrop/
///     ...
/// </code>
/// </summary>
public interface IDatasetLoader
{
    /// <summary>
    /// Загружает и перемешивает все образцы из rootDir.
    /// Возвращает кортеж (trainSamples, valSamples).
    /// </summary>
    Task<(IReadOnlyList<ImageSample> Train, IReadOnlyList<ImageSample> Validation)>
        LoadAsync(string rootDir, float validationSplit, CancellationToken ct = default);

    /// <summary>
    /// Нарезает список образцов на батчи заданного размера.
    /// </summary>
    IEnumerable<TrainingBatch> GetBatches(
        IReadOnlyList<ImageSample> samples,
        int batchSize,
        bool shuffle = true);
}
