using Microsoft.Extensions.DependencyInjection;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Preprocessing;
using OpenCvSharp;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace MobileNetV3.UI;

public partial class MainForm : Form
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IImagePreprocessor _preprocessor;
    private readonly IDatasetLoader _datasetLoader;
    private readonly TrainingConfig _config;

    // Layout
    private TabControl _mainTabControl = null!;
    private TabPage _trainingTab = null!;
    private TabPage _imageTestTab = null!;
    private TabPage _videoTestTab = null!;

    // ── Training controls ────────────────────────────────────────────────
    private TextBox _txtDatasetPath = null!;
    private Label _lblDatasetInfo = null!;
    private Button _btnSelectDataset = null!;
    private Button _btnStartTraining = null!;
    private ProgressBar _trainingProgress = null!;
    private TextBox _txtTrainingLog = null!;
    private CancellationTokenSource? _trainingCts;

    // ── Image test controls ──────────────────────────────────────────────
    private PictureBox _imagePreview = null!;
    private Button _btnSelectImage = null!;
    private Button _btnPredictImage = null!;
    private ListBox _lstImageResults = null!;
    private Label _lblImagePrediction = null!;
    private string? _currentImagePath;

    // ── Video test controls ──────────────────────────────────────────────
    private PictureBox _videoPreview = null!;
    private Button _btnSelectVideo = null!;
    private Button _btnStartVideo = null!;
    private Button _btnStopVideo = null!;
    private Button _btnStartWebcam = null!;
    private ComboBox _cmbVideoSource = null!;
    private ListBox _lstVideoResults = null!;
    private Label _lblVideoStatus = null!;
    private CancellationTokenSource? _videoCts;
    private string? _currentVideoPath;
    private bool _isRunning => _videoCts is { IsCancellationRequested: false };

    public MainForm(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
        _preprocessor = serviceProvider.GetRequiredService<IImagePreprocessor>();
        _datasetLoader = serviceProvider.GetRequiredService<IDatasetLoader>();
        _config = serviceProvider.GetRequiredService<TrainingConfig>();

        BuildLayout();
        BuildTrainingTab();
        BuildImageTestTab();
        BuildVideoTestTab();

        Text = "MobileNetV3 — Training & Testing";
        Size = new Size(1200, 800);
        MinimumSize = new Size(1000, 700);
        StartPosition = FormStartPosition.CenterScreen;
    }

    // ── Root layout ──────────────────────────────────────────────────────
    private void BuildLayout()
    {
        _mainTabControl = new TabControl
        {
            Dock = DockStyle.Fill,
            Font = new Font("Segoe UI", 10F)
        };

        _trainingTab = new TabPage("⚙  Training");
        _imageTestTab = new TabPage("🖼  Image Testing");
        _videoTestTab = new TabPage("🎬  Video Testing");

        _mainTabControl.TabPages.AddRange(
            new TabPage[] { _trainingTab, _imageTestTab, _videoTestTab });

        Controls.Add(_mainTabControl);
    }
}
