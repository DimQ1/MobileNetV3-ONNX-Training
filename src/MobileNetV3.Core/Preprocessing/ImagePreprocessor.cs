using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using OpenCvSharp;

namespace MobileNetV3.Core.Preprocessing;

/// <summary>
/// Препроцессор изображений на базе OpenCvSharp.
/// Выполняет: загрузку → масштабирование → BGR→RGB → нормализацию → CHW layout.
/// </summary>
public sealed class ImagePreprocessor : IImagePreprocessor
{
    private readonly TrainingConfig _config;
    private readonly ILogger<ImagePreprocessor> _logger;

    // Границы для случайной обрезки при аугментации (10%)
    private const float CropRatio = 0.9f;

    private static readonly Random Rng = new(42);

    public ImagePreprocessor(TrainingConfig config, ILogger<ImagePreprocessor> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <inheritdoc/>
    public float[] Preprocess(string filePath, bool augment = false)
    {
        using var mat = Cv2.ImRead(filePath, ImreadModes.Color);

        if (mat.Empty())
            throw new InvalidOperationException($"Не удалось загрузить изображение: {filePath}");

        return PreprocessMat(mat, augment);
    }

    /// <inheritdoc/>
    public float[] PreprocessMat(Mat image, bool augment = false)
    {
        using var processed = image.Clone();

        if (augment && _config.UseAugmentation)
            ApplyAugmentation(processed);

        // 1. Масштабирование до целевого размера
        using var resized = new Mat();
        Cv2.Resize(processed, resized, new Size(_config.ImageSize, _config.ImageSize),
            interpolation: InterpolationFlags.Linear);

        // 2. Конвертация BGR (OpenCV default) → RGB
        using var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        // 3. Конвертация в float32 и нормализация [0..255] → [0..1]
        using var floatMat = new Mat();
        rgb.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

        // 4. Применение ImageNet-нормализации: (x - mean) / std по каналам
        return ConvertToNormalizedCHW(floatMat);
    }

    /// <summary>
    /// Переводит HWC float Mat в плоский CHW массив с нормализацией по каналам.
    /// </summary>
    private float[] ConvertToNormalizedCHW(Mat floatMat)
    {
        int h = floatMat.Rows;
        int w = floatMat.Cols;
        int channelSize = h * w;
        var result = new float[3 * channelSize];

        float meanR = _config.NormMean[0], meanG = _config.NormMean[1], meanB = _config.NormMean[2];
        float stdR  = _config.NormStd[0],  stdG  = _config.NormStd[1],  stdB  = _config.NormStd[2];

        // Извлекаем данные из Mat напрямую через индексер
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var pixel = floatMat.At<Vec3f>(y, x); // (R, G, B) после конвертации

                int idx = y * w + x;
                result[0 * channelSize + idx] = (pixel.Item0 - meanR) / stdR; // R
                result[1 * channelSize + idx] = (pixel.Item1 - meanG) / stdG; // G
                result[2 * channelSize + idx] = (pixel.Item2 - meanB) / stdB; // B
            }
        }

        return result;
    }

    /// <summary>
    /// Применяет случайные аугментации к Mat (in-place).
    /// </summary>
    private static void ApplyAugmentation(Mat mat)
    {
        // Случайное горизонтальное отражение (50%)
        if (Rng.NextDouble() > 0.5)
            Cv2.Flip(mat, mat, FlipMode.Y);

        // Случайная обрезка с последующим масштабированием
        if (Rng.NextDouble() > 0.5)
        {
            int h = mat.Rows, w = mat.Cols;
            int cropH = (int)(h * CropRatio);
            int cropW = (int)(w * CropRatio);
            int offsetY = Rng.Next(0, h - cropH);
            int offsetX = Rng.Next(0, w - cropW);

            using var cropped = new Mat(mat, new Rect(offsetX, offsetY, cropW, cropH));
            Cv2.Resize(cropped, mat, new Size(w, h));
        }

        // Случайное изменение яркости/контраста
        if (Rng.NextDouble() > 0.5)
        {
            double alpha = 0.8 + Rng.NextDouble() * 0.4; // [0.8, 1.2]
            int beta = (int)(Rng.NextDouble() * 30 - 15);  // [-15, 15]
            mat.ConvertTo(mat, -1, alpha, beta);
        }

        // Случайное вращение на небольшой угол [-15°, +15°]
        if (Rng.NextDouble() > 0.7)
        {
            double angle = Rng.NextDouble() * 30 - 15;
            var center = new Point2f(mat.Cols / 2f, mat.Rows / 2f);
            using var rotMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);
            Cv2.WarpAffine(mat, mat, rotMatrix, mat.Size(),
                flags: InterpolationFlags.Linear,
                borderMode: BorderTypes.Reflect);
        }
    }
}
