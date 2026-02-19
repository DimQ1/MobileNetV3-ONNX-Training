using Microsoft.Extensions.Logging.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Preprocessing;
using OpenCvSharp;

namespace MobileNetV3.Tests.Preprocessing;

public sealed class ImagePreprocessorTests : IDisposable
{
    private readonly TrainingConfig _config;
    private readonly ImagePreprocessor _preprocessor;
    private readonly string _tempDir;

    public ImagePreprocessorTests()
    {
        _config = new TrainingConfig { ImageSize = 224 };
        _preprocessor = new ImagePreprocessor(
            _config,
            NullLogger<ImagePreprocessor>.Instance);

        _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);
    }

    // ─── PreprocessMat ────────────────────────────────────────────────────────

    [Fact]
    public void PreprocessMat_ReturnsCorrectTensorLength()
    {
        using var mat = CreateTestMat(300, 400); // Нестандартный размер входа

        var result = _preprocessor.PreprocessMat(mat);

        // Ожидаем CHW: 3 * 224 * 224
        Assert.Equal(3 * 224 * 224, result.Length);
    }

    [Fact]
    public void PreprocessMat_OutputValuesAreNormalized()
    {
        // Белое изображение (255, 255, 255 во всех каналах BGR)
        using var whiteMat = new Mat(new Size(224, 224), MatType.CV_8UC3, new Scalar(255, 255, 255));

        var result = _preprocessor.PreprocessMat(whiteMat);

        // После нормализации белый пиксель:
        // R: (1.0 - 0.485) / 0.229 ≈  2.2489
        // G: (1.0 - 0.456) / 0.224 ≈  2.4286
        // B: (1.0 - 0.406) / 0.225 ≈  2.6400
        float expectedR = (1.0f - 0.485f) / 0.229f;
        Assert.Equal(expectedR, result[0], precision: 3);
    }

    [Fact]
    public void PreprocessMat_BlackImage_GivesNegativeValues()
    {
        // Чёрное изображение (0, 0, 0) даёт отрицательные значения после ImageNet нормализации
        using var blackMat = new Mat(new Size(224, 224), MatType.CV_8UC3, new Scalar(0, 0, 0));

        var result = _preprocessor.PreprocessMat(blackMat);

        // (0.0 - 0.485) / 0.229 ≈ -2.118  (канал R)
        Assert.True(result[0] < 0, "Нормализованное значение чёрного должно быть отрицательным");
    }

    [Fact]
    public void PreprocessMat_DifferentInputSizes_ProduceSameTensorShape()
    {
        var sizes = new[] { (100, 100), (640, 480), (1920, 1080) };

        foreach (var (w, h) in sizes)
        {
            using var mat = CreateTestMat(h, w);
            var result = _preprocessor.PreprocessMat(mat);

            Assert.Equal(3 * 224 * 224, result.Length);
        }
    }

    [Fact]
    public void PreprocessMat_WithAugmentation_StillReturnsCorrectShape()
    {
        using var mat = CreateTestMat(224, 224);

        var result = _preprocessor.PreprocessMat(mat, augment: true);

        Assert.Equal(3 * 224 * 224, result.Length);
    }

    // ─── Preprocess (file) ────────────────────────────────────────────────────

    [Fact]
    public void Preprocess_ValidFile_ReturnsCorrectTensor()
    {
        string filePath = CreateTestImageFile(224, 224);

        var result = _preprocessor.Preprocess(filePath);

        Assert.Equal(3 * 224 * 224, result.Length);
    }

    [Fact]
    public void Preprocess_NonExistentFile_ThrowsInvalidOperation()
    {
        Assert.Throws<InvalidOperationException>(
            () => _preprocessor.Preprocess("nonexistent_file.jpg"));
    }

    // ─── Нормализация ─────────────────────────────────────────────────────────

    [Fact]
    public void PreprocessMat_ChannelOrderIsRGB_NotBGR()
    {
        // Создаём изображение с ярким красным каналом (B=0, G=0, R=255 в BGR)
        using var mat = new Mat(new Size(224, 224), MatType.CV_8UC3, new Scalar(0, 0, 255));

        var result = _preprocessor.PreprocessMat(mat);

        int channelSize = 224 * 224;

        // После BGR→RGB первый канал должен быть ярким (R=255→1.0)
        float rChannelVal = result[0]; // первый пиксель, канал R

        // G и B каналы должны быть нулём до нормализации
        float gChannelVal = result[channelSize];       // G канал, первый пиксель
        float bChannelVal = result[2 * channelSize];   // B канал, первый пиксель

        // R должен быть положительным и большим (нормализованный 1.0)
        Assert.True(rChannelVal > 2.0f, $"R-канал ожидается ~2.25, получено: {rChannelVal}");
        // G и B должны быть отрицательными (нормализованный 0.0)
        Assert.True(gChannelVal < 0, $"G-канал ожидается отрицательным, получено: {gChannelVal}");
        Assert.True(bChannelVal < 0, $"B-канал ожидается отрицательным, получено: {bChannelVal}");
    }

    // ─── Вспомогательные методы ───────────────────────────────────────────────

    private static Mat CreateTestMat(int rows, int cols)
    {
        var mat = new Mat(new Size(cols, rows), MatType.CV_8UC3);
        var rng = new RNG(42);
        rng.Fill(mat, DistributionType.Uniform, 0, 255);
        return mat;
    }

    private string CreateTestImageFile(int rows, int cols)
    {
        using var mat = CreateTestMat(rows, cols);
        string path = Path.Combine(_tempDir, $"test_{Guid.NewGuid()}.jpg");
        Cv2.ImWrite(path, mat);
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}
