using MobileNetV3.Core.Models;
using OpenCvSharp;

namespace MobileNetV3.Core.Abstractions;

/// <summary>
/// Контракт для препроцессора изображений.
/// Принимает Mat (OpenCV) и возвращает нормализованный тензор float CHW.
/// </summary>
public interface IImagePreprocessor
{
    /// <summary>
    /// Загружает изображение с диска, масштабирует до targetSize×targetSize,
    /// конвертирует BGR→RGB, нормализует и переводит в формат CHW.
    /// </summary>
    /// <param name="filePath">Путь к файлу изображения</param>
    /// <param name="augment">Применять ли аугментацию (только для train)</param>
    /// <returns>Плоский массив float [C * H * W]</returns>
    float[] Preprocess(string filePath, bool augment = false);

    /// <summary>
    /// Препроцессинг уже загруженного Mat-объекта.
    /// </summary>
    float[] PreprocessMat(Mat image, bool augment = false);
}
