using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Export;
using MobileNetV3.Core.Metrics;
using MobileNetV3.Core.Models;
using MobileNetV3.Core.Preprocessing;
using MobileNetV3.Core.Training;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OcvPoint = OpenCvSharp.Point;
using OcvSize = OpenCvSharp.Size;
using OcvWindow = OpenCvSharp.Window;

namespace MobileNetV3.UI;

public partial class MainWindow : System.Windows.Window
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IImagePreprocessor _preprocessor;
    private readonly IDatasetLoader _datasetLoader;
    private readonly TrainingConfig _config;

    // Training
    private CancellationTokenSource? _trainingCts;

    // Image
    private string? _currentImagePath;

    // Video
    private CancellationTokenSource? _videoCts;
    private BitmapSource? _currentVideoFrame;
    private string? _currentVideoPath;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
        Closing += MainWindow_Closing;
        _serviceProvider = ((App)Application.Current).ServiceProvider;
        _preprocessor = _serviceProvider.GetRequiredService<IImagePreprocessor>();
        _datasetLoader = _serviceProvider.GetRequiredService<IDatasetLoader>();
        _config = _serviceProvider.GetRequiredService<TrainingConfig>();
    }

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        BtnSelectDataset.Click += OnSelectDataset;
        BtnStartTraining.Click += OnStartTraining;
        BtnSelectImage.Click += OnSelectImage;
        BtnPredictImage.Click += OnPredictImage;
        BtnSelectVideo.Click += OnSelectVideo;
        BtnStartWebcam.Click += OnStartWebcam;
        BtnStartVideo.Click += OnStartVideo;
        BtnStopVideo.Click += OnStopVideo;
    }

    private void MainWindow_Closing(object? sender, CancelEventArgs e)
    {
        _trainingCts?.Cancel();
        _videoCts?.Cancel();
        ImagePreview.Source = null;
        VideoPreview.Source = null;
    }

    // ── Training ──────────────────────────────────────────────────────────
    private async void OnSelectDataset(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            ValidateNames = false,
            CheckFileExists = false,
            CheckPathExists = true,
            FileName = "Folder Selection."
        };
        if (dlg.ShowDialog() != true) return;

        var path = System.IO.Path.GetDirectoryName(dlg.FileName)!;
        TxtDatasetPath.Text = path;
        RefreshDatasetInfo(path);
    }

    private void RefreshDatasetInfo(string path)
    {
        try
        {
            var dirs = Directory.GetDirectories(path);
            var imgs = dirs.Sum(d => Directory.GetFiles(d, "*.jpg").Length +
                                      Directory.GetFiles(d, "*.jpeg").Length +
                                      Directory.GetFiles(d, "*.png").Length +
                                      Directory.GetFiles(d, "*.bmp").Length);

            var classNames = string.Join(", ", dirs.Select(System.IO.Path.GetFileName));
            LblDatasetInfo.Content = $"Classes: {dirs.Length} | Images: {imgs} [{classNames}]";
            LblDatasetInfo.Foreground = dirs.Length > 0 ? System.Windows.Media.Brushes.Green : System.Windows.Media.Brushes.Red;
        }
        catch (Exception ex)
        {
            LblDatasetInfo.Content = $"Error: {ex.Message}";
            LblDatasetInfo.Foreground = System.Windows.Media.Brushes.Red;
        }
    }

    private async void OnStartTraining(object sender, RoutedEventArgs e)
    {
        if (!Directory.Exists(TxtDatasetPath.Text))
        {
            MessageBox.Show("Please select a valid dataset directory.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (BtnStartTraining.Tag?.ToString() == "running")
            StopTraining();
        else
            _ = StartTrainingAsync();
    }

    private async Task StartTrainingAsync()
    {
        BtnStartTraining.Content = "⏹ Stop Training";
        BtnStartTraining.Background = System.Windows.Media.Brushes.DarkRed;
        BtnStartTraining.Tag = "running";
        TrainingProgress.Value = 0;
        TxtTrainingLog.Text = string.Empty;
        _trainingCts = new CancellationTokenSource();

        TrainingReport? report = null;

        try
        {
            Dispatcher.Invoke(() => AppendLog("Loading dataset…"));

            var datasetPath = TxtDatasetPath.Text;
            var ct = _trainingCts.Token;

            var (trainSamples, valSamples) = await Task.Run(() =>
                _datasetLoader.LoadAsync(datasetPath, _config.ValidationSplit, ct).Result, ct);

            Dispatcher.Invoke(() => AppendLog($"Train: {trainSamples.Count} samples | Validation: {valSamples.Count} samples"));
            Dispatcher.Invoke(() => AppendLog($"Starting training for {_config.Epochs} epochs…"));

            var progress = new Progress<EpochResult>(r =>
            {
                var pct = (int)((float)r.Epoch / _config.Epochs * 100);
                TrainingProgress.Value = Math.Min(pct, 100);

                Dispatcher.Invoke(() => AppendLog(r.ToString()));
            });

            report = await Task.Run(() =>
            {
                using var trainer = BuildTrainer(_config);
                return trainer.TrainAsync(trainSamples, valSamples, progress, ct).Result;
            }, ct);

            TrainingProgress.Value = 100;
            Dispatcher.Invoke(() => AppendLog($"Training complete! Best accuracy: {report.BestValidationAccuracy:P2} (epoch {report.BestEpoch})"));
            Dispatcher.Invoke(() => AppendLog($"Model saved → {report.ModelOutputPath}"));
        }
        catch (OperationCanceledException)
        {
            Dispatcher.Invoke(() => AppendLog("Training cancelled."));
        }
        catch (Exception ex)
        {
            Dispatcher.Invoke(() => AppendLog($"ERROR: {ex.Message}"));
        }
        finally
        {
            Dispatcher.Invoke(() =>
            {
                BtnStartTraining.Content = "▶ Start Training";
                BtnStartTraining.Background = System.Windows.Media.Brushes.DarkGreen;
                BtnStartTraining.Tag = null;
            });
            _trainingCts?.Dispose();
            _trainingCts = null;
        }
    }

    private void StopTraining()
    {
        _trainingCts?.Cancel();
        AppendLog("Stop signal sent — finishing current epoch…");
    }

    private void AppendLog(string message)
    {
        var ts = DateTime.Now.ToString("HH:mm:ss");
        TxtTrainingLog.Text += $"[{ts}] {message}{Environment.NewLine}";
        TxtTrainingLog.ScrollToEnd();
    }

    private IModelTrainer BuildTrainer(TrainingConfig cfg)
    {
        var loggerFactory = LoggerFactory.Create(builder => builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information));
        var trainerLogger = loggerFactory.CreateLogger<ModelTrainer>();
        var schedulerLogger = loggerFactory.CreateLogger<LearningRateScheduler>();
        return new ModelTrainer(cfg, _datasetLoader, new MetricsTracker(), new MetricsTracker(), new LearningRateScheduler(cfg, schedulerLogger), trainerLogger);
    }

    // ── Image Testing ─────────────────────────────────────────────────────
    private void OnSelectImage(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select image",
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        _currentImagePath = dlg.FileName;
        ImagePreview.Source = new BitmapImage(new Uri(_currentImagePath));
        LstImageResults.Items.Clear();
        LblImagePrediction.Content = "Ready - press Predict";
        LblImagePrediction.Foreground = System.Windows.Media.Brushes.White;
    }

    private async void OnPredictImage(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_currentImagePath))
        {
            MessageBox.Show("Please open an image first.", "No image");
            return;
        }

        BtnPredictImage.IsEnabled = false;
        LblImagePrediction.Content = "Running...";
        LblImagePrediction.Foreground = System.Windows.Media.Brushes.Yellow;

        try
        {
            var (top, scores) = await Task.Run(() => RunImageInference(_currentImagePath!));

            LstImageResults.Items.Clear();
            foreach (var (cls, conf) in scores.OrderByDescending(x => x.Conf))
                LstImageResults.Items.Add($"{cls,-18} {conf * 100:F2} %");

            LblImagePrediction.Content = $"{top.Class}  ({top.Conf * 100:F1} %)";
            LblImagePrediction.Foreground = top.Conf >= 0.70f ? System.Windows.Media.Brushes.LimeGreen : System.Windows.Media.Brushes.Orange;
        }
        catch (FileNotFoundException)
        {
            LblImagePrediction.Content = "Model not found - train first!";
            LblImagePrediction.Foreground = System.Windows.Media.Brushes.OrangeRed;
        }
        catch (Exception ex)
        {
            LblImagePrediction.Content = "Error: " + ex.Message;
            LblImagePrediction.Foreground = System.Windows.Media.Brushes.Red;
        }
        finally
        {
            BtnPredictImage.IsEnabled = true;
        }
    }

    private ((string Class, float Conf) Top, List<(string Class, float Conf)> All) RunImageInference(string imagePath)
    {
        var modelPath = Path.Combine("models", "mobilenet_v3.onnx");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model not found", modelPath);

        var metaPath = Path.Combine("models", "mobilenet_v3.json");
        var meta = ModelInfo.Load(metaPath);

        using var mat = Cv2.ImRead(imagePath);
        if (mat.Empty()) throw new Exception("Cannot read image file.");

        var floatData = _preprocessor.PreprocessMat(mat);
        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        floatData.CopyTo(tensor.Buffer.Span);
        var scores = RunOnnxSession(modelPath, tensor, meta.ClassNames);
        var top = scores.OrderByDescending(x => x.Conf).First();
        return (top, scores);
    }

    private static List<(string Class, float Conf)> RunOnnxSession(string modelPath, DenseTensor<float> tensor, string[] classNames)
    {
        using var session = new InferenceSession(modelPath);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
        using var results = session.Run(inputs);

        var raw = results.First().AsTensor<float>().ToArray();
        var softmax = Softmax(raw);
        return classNames.Zip(softmax, (c, s) => (c, s)).ToList();
    }

    private static float[] Softmax(float[] x)
    {
        var max = x.Max();
        var exp = x.Select(v => (float)Math.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    // ── Video Testing ─────────────────────────────────────────────────────
    private void OnSelectVideo(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select video file",
            Filter = "Video files|*.mp4;*.avi;*.mov;*.wmv;*.mkv|All files|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        _currentVideoPath = dlg.FileName;
        CmbVideoSource.SelectedIndex = -1;
        LblVideoStatus.Content = "Loaded: " + System.IO.Path.GetFileName(_currentVideoPath);
        LblVideoStatus.Foreground = System.Windows.Media.Brushes.CornflowerBlue;
    }

    private async void OnStartWebcam(object sender, RoutedEventArgs e)
    {
        int camId = CmbVideoSource.SelectedIndex;
        await RunStreamAsync(camId.ToString(), true);
    }

    private async void OnStartVideo(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_currentVideoPath))
        {
            MessageBox.Show("Please select a video file first.", "No video");
            return;
        }
        await RunStreamAsync(_currentVideoPath!, false);
    }

    private void OnStopVideo(object sender, RoutedEventArgs e)
    {
        _videoCts?.Cancel();
        LblVideoStatus.Content = "Status: Stopping...";
        LblVideoStatus.Foreground = System.Windows.Media.Brushes.Orange;
    }

    private async Task RunStreamAsync(string source, bool isWebcam)
    {
        BtnStartVideo.IsEnabled = false;
        BtnStartWebcam.IsEnabled = false;
        BtnStopVideo.IsEnabled = true;
        _videoCts = new CancellationTokenSource();

        LblVideoStatus.Content = isWebcam ? "Status: Webcam running" : "Status: Video playing";
        LblVideoStatus.Foreground = System.Windows.Media.Brushes.LimeGreen;

        try
        {
            var ct = _videoCts.Token;
            await Task.Run(() => ProcessStream(source, isWebcam, ct), ct);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Dispatcher.Invoke(() => MessageBox.Show("Stream error: " + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error));
        }
        finally
        {
            _videoCts?.Dispose();
            _videoCts = null;
            Dispatcher.Invoke(() =>
            {
                BtnStartVideo.IsEnabled = true;
                BtnStartWebcam.IsEnabled = true;
                BtnStopVideo.IsEnabled = false;
                LblVideoStatus.Content = "Status: Idle";
                LblVideoStatus.Foreground = System.Windows.Media.Brushes.Gray;
            });
        }
    }

    private void ProcessStream(string source, bool isWebcam, CancellationToken ct)
    {
        var modelPath = Path.Combine("models", "mobilenet_v3.onnx");
        var metaPath = Path.Combine("models", "mobilenet_v3.json");
        var modelReady = File.Exists(modelPath) && File.Exists(metaPath);

        string[]? classNames = null;
        InferenceSession? session = null;

        if (modelReady)
        {
            var meta = ModelInfo.Load(metaPath);
            classNames = meta.ClassNames;
            session = new InferenceSession(modelPath);
        }

        int camId = int.TryParse(source, out var id) ? id : 0;
        using var cap = isWebcam ? new VideoCapture(camId) : new VideoCapture(source);

        if (!cap.IsOpened())
            throw new Exception(isWebcam ? "Cannot open camera " + camId : "Cannot open video file.");

        int frameIdx = 0;
        var lastInfer = DateTime.MinValue;
        string lastLabel = "—";
        float lastConf = 0f;

        using (session)
        {
            while (!ct.IsCancellationRequested)
            {
                using var frame = new Mat();
                if (!cap.Read(frame) || frame.Empty()) break;

                frameIdx++;

                if (modelReady && session != null &&
                    (DateTime.Now - lastInfer).TotalMilliseconds >= 500)
                {
                    var floatData = _preprocessor.PreprocessMat(frame);
                    var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                    floatData.CopyTo(tensor.Buffer.Span);
                    var inputs = new[] { NamedOnnxValue.CreateFromTensor<float>("input", tensor) };
                    using var res = session.Run(inputs);

                    var raw = res.First().AsTensor<float>().ToArray();
                    var softmax = Softmax(raw);
                    var best = softmax.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();

                    lastLabel = classNames![best.i];
                    lastConf = best.v;
                    lastInfer = DateTime.Now;

                    Dispatcher.Invoke(() =>
                    {
                        LstVideoResults.Items.Insert(0, string.Format("[{0}] {1}  {2:F1}%",
                            DateTime.Now.ToString("HH:mm:ss"), lastLabel, lastConf * 100));
                        if (LstVideoResults.Items.Count > 60)
                            LstVideoResults.Items.RemoveAt(LstVideoResults.Items.Count - 1);
                    });
                }

                DrawOverlay(frame, lastLabel, lastConf, frameIdx, modelReady);

                var bitmap = frame.ToBitmap();
                var bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    bitmap.GetHbitmap(),
                    IntPtr.Zero,
                    new System.Windows.Int32Rect(0, 0, bitmap.Width, bitmap.Height),
                    System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
                bs.Freeze();
                Dispatcher.Invoke(() =>
                {
                    _currentVideoFrame = bs;
                    VideoPreview.Source = bs;
                });

                Thread.Sleep(33);
            }
        }
    }

    private static void DrawOverlay(Mat frame, string label, float conf, int frameIdx, bool modelReady)
    {
        var fnt = HersheyFonts.HersheySimplex;
        var scale = 0.9f;
        var thick = 2;

        Cv2.PutText(frame, "Frame: " + frameIdx, new OcvPoint(10, frame.Rows - 10), fnt, 0.55, new Scalar(180, 180, 180), 1);

        if (!modelReady)
        {
            Cv2.PutText(frame, "No model - train first", new OcvPoint(10, 35), fnt, scale, new Scalar(0, 100, 255), thick);
            return;
        }

        var text = string.Format("{0}  {1:F1}%", label, conf * 100);
        var color = conf >= 0.70 ? new Scalar(0, 230, 0) : new Scalar(0, 165, 255);
        var txtSize = Cv2.GetTextSize(text, fnt, scale, thick, out _);

        Cv2.Rectangle(frame, new OcvPoint(6, 6), new OcvPoint((int)txtSize.Width + 14, (int)txtSize.Height + 14), new Scalar(0, 0, 0, 160), -1);
        Cv2.PutText(frame, text, new OcvPoint(10, (int)txtSize.Height + 10), fnt, scale, color, thick);
    }
}


