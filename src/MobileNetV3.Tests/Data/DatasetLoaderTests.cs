using Microsoft.Extensions.Logging.Abstractions;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Models;
using MobileNetV3.Core.Preprocessing;
using OpenCvSharp;

namespace MobileNetV3.Tests.Data;

public sealed class DatasetLoaderTests : IDisposable
{
    private readonly string _tempDir;
    private readonly TrainingConfig _config;
    private readonly IDatasetLoader _loader;

    public DatasetLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);

        _config = new TrainingConfig
        {
            ImageSize    = 224,
            ClassLabels  = ["Clean", "DryDirt", "WaterDrop"]
        };

        IImagePreprocessor preprocessor = new ImagePreprocessor(
            _config,
            NullLogger<ImagePreprocessor>.Instance);

        _loader = new DatasetLoader(
            _config,
            preprocessor,
            NullLogger<DatasetLoader>.Instance);
    }

    // ─── LoadAsync ────────────────────────────────────────────────────────────

    [Fact]
    public async Task LoadAsync_ValidDataset_ReturnsSamplesForAllClasses()
    {
        CreateFakeDataset(imagesPerClass: 10);

        var (train, val) = await _loader.LoadAsync(_tempDir, validationSplit: 0.2f);

        Assert.NotEmpty(train);
        Assert.NotEmpty(val);
        // 3 класса * 10 изображений = 30, 20% val = 6
        Assert.Equal(30, train.Count + val.Count);
    }

    [Fact]
    public async Task LoadAsync_StratifiedSplit_PreservesClassDistribution()
    {
        CreateFakeDataset(imagesPerClass: 20);

        var (train, val) = await _loader.LoadAsync(_tempDir, validationSplit: 0.2f);

        // Каждый класс должен присутствовать в обеих выборках
        var trainClasses = train.Select(s => s.ClassName).Distinct().ToHashSet();
        var valClasses   = val.Select(s => s.ClassName).Distinct().ToHashSet();

        Assert.Equal(3, trainClasses.Count);
        Assert.Equal(3, valClasses.Count);
    }

    [Fact]
    public async Task LoadAsync_SamplesHaveCorrectLabelIndices()
    {
        CreateFakeDataset(imagesPerClass: 5);

        var (train, _) = await _loader.LoadAsync(_tempDir, validationSplit: 0.2f);

        foreach (var sample in train)
        {
            int expectedLabel = Array.IndexOf(_config.ClassLabels, sample.ClassName);
            Assert.Equal(expectedLabel, (int)sample.Label);
        }
    }

    [Fact]
    public async Task LoadAsync_NonExistentDir_ThrowsDirectoryNotFound()
    {
        await Assert.ThrowsAsync<DirectoryNotFoundException>(
            () => _loader.LoadAsync("/non/existent/path", 0.2f));
    }

    [Fact]
    public async Task LoadAsync_EmptyDir_ThrowsInvalidOperation()
    {
        // Директория есть, но папок классов нет
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _loader.LoadAsync(_tempDir, 0.2f));
    }

    [Fact]
    public async Task LoadAsync_SampleTensorHasCorrectDimensions()
    {
        CreateFakeDataset(imagesPerClass: 3);

        var (train, _) = await _loader.LoadAsync(_tempDir, validationSplit: 0.3f);

        int expectedTensorSize = 3 * _config.ImageSize * _config.ImageSize;

        foreach (var sample in train)
            Assert.Equal(expectedTensorSize, sample.Tensor.Length);
    }

    // ─── GetBatches ───────────────────────────────────────────────────────────

    [Fact]
    public void GetBatches_ReturnsCorrectNumberOfBatches()
    {
        var samples = CreateFakeSamples(100);

        var batches = _loader.GetBatches(samples, batchSize: 16).ToList();

        // 100 / 16 = 6 полных + 1 остаток = 7 батчей
        Assert.Equal(7, batches.Count);
    }

    [Fact]
    public void GetBatches_LastBatchMayBeSmaller()
    {
        var samples = CreateFakeSamples(35);

        var batches = _loader.GetBatches(samples, batchSize: 16).ToList();

        Assert.Equal(3, batches.Count);
        Assert.Equal(16, batches[0].BatchSize);
        Assert.Equal(16, batches[1].BatchSize);
        Assert.Equal(3, batches[2].BatchSize);  // остаток
    }

    [Fact]
    public void GetBatches_BatchImagesHaveCorrectShape()
    {
        var samples = CreateFakeSamples(10);

        var batch = _loader.GetBatches(samples, batchSize: 4).First();

        // NCHW: 4 * 3 * 224 * 224
        Assert.Equal(4 * 3 * _config.ImageSize * _config.ImageSize, batch.Images.Length);
    }

    [Fact]
    public void GetBatches_BatchLabelsMatchSampleCount()
    {
        var samples = CreateFakeSamples(10);

        foreach (var batch in _loader.GetBatches(samples, batchSize: 4))
            Assert.Equal(batch.BatchSize, batch.Labels.Length);
    }

    [Fact]
    public void GetBatches_TotalSamplesAcrossBatchesEqualsInput()
    {
        var samples = CreateFakeSamples(47);

        int totalProcessed = _loader
            .GetBatches(samples, batchSize: 8)
            .Sum(b => b.BatchSize);

        Assert.Equal(47, totalProcessed);
    }

    // ─── Вспомогательные методы ───────────────────────────────────────────────

    private void CreateFakeDataset(int imagesPerClass)
    {
        foreach (var cls in _config.ClassLabels)
        {
            string classDir = Path.Combine(_tempDir, cls);
            Directory.CreateDirectory(classDir);

            for (int i = 0; i < imagesPerClass; i++)
            {
                using var mat = new Mat(new Size(224, 224), MatType.CV_8UC3,
                    new Scalar(i * 8 % 255, i * 4 % 255, i * 2 % 255));
                Cv2.ImWrite(Path.Combine(classDir, $"img_{i:D4}.jpg"), mat);
            }
        }
    }

    private List<ImageSample> CreateFakeSamples(int count)
    {
        return Enumerable.Range(0, count).Select(i => new ImageSample
        {
            Tensor    = new float[3 * _config.ImageSize * _config.ImageSize],
            Label     = i % _config.NumClasses,
            FilePath  = $"fake_{i}.jpg",
            ClassName = _config.ClassLabels[i % _config.NumClasses]
        }).ToList();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}
