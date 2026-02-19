using Microsoft.Extensions.Logging.Abstractions;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Metrics;
using MobileNetV3.Core.Models;
using MobileNetV3.Core.Training;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace MobileNetV3.UI;

public partial class MainForm
{
    // ── Build UI ──────────────────────────────────────────────────────────
    private void BuildTrainingTab()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 3,
            ColumnCount = 1,
            Padding = new Padding(12)
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 110));  // dataset group
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 110));  // controls group
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));  // log group

        root.Controls.Add(BuildDatasetGroup(), 0, 0);
        root.Controls.Add(BuildControlsGroup(), 0, 1);
        root.Controls.Add(BuildLogGroup(), 0, 2);

        _trainingTab.Controls.Add(root);
    }

    private GroupBox BuildDatasetGroup()
    {
        var grp = new GroupBox
        {
            Text = "Dataset",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };

        _txtDatasetPath = new TextBox
        {
            Location = new Point(10, 24),
            Size = new Size(560, 26),
            ReadOnly = true,
            Font = new Font("Segoe UI", 9.5F)
        };

        _btnSelectDataset = MakeButton("Browse…", Color.FromArgb(0, 120, 215));
        _btnSelectDataset.Location = new Point(580, 22);
        _btnSelectDataset.Size = new Size(100, 30);
        _btnSelectDataset.Click += OnSelectDataset;

        _lblDatasetInfo = new Label
        {
            Text = "No dataset selected",
            Location = new Point(10, 60),
            AutoSize = true,
            ForeColor = Color.Gray
        };

        grp.Controls.AddRange(new Control[]
            { _txtDatasetPath, _btnSelectDataset, _lblDatasetInfo });
        return grp;
    }

    private GroupBox BuildControlsGroup()
    {
        var grp = new GroupBox
        {
            Text = "Training",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };

        _btnStartTraining = MakeButton("▶  Start Training", Color.FromArgb(0, 150, 0));
        _btnStartTraining.Location = new Point(10, 22);
        _btnStartTraining.Size = new Size(170, 34);
        _btnStartTraining.Click += OnStartTraining;

        _trainingProgress = new ProgressBar
        {
            Location = new Point(10, 66),
            Size = new Size(660, 22),
            Minimum = 0,
            Maximum = 100,
            Style = ProgressBarStyle.Continuous
        };

        grp.Controls.AddRange(new Control[] { _btnStartTraining, _trainingProgress });
        return grp;
    }

    private GroupBox BuildLogGroup()
    {
        var grp = new GroupBox
        {
            Text = "Training Log",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };

        _txtTrainingLog = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            Font = new Font("Consolas", 9F),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.LimeGreen
        };

        grp.Controls.Add(_txtTrainingLog);
        return grp;
    }

    // ── Event Handlers ────────────────────────────────────────────────────
    private void OnSelectDataset(object? sender, EventArgs e)
    {
        using var dlg = new FolderBrowserDialog
        {
            Description = "Select dataset root directory (sub-folders = class names)",
            UseDescriptionForTitle = true
        };

        if (dlg.ShowDialog() != DialogResult.OK) return;

        _txtDatasetPath.Text = dlg.SelectedPath;
        RefreshDatasetInfo(dlg.SelectedPath);
    }

    private void RefreshDatasetInfo(string path)
    {
        try
        {
            var dirs = Directory.GetDirectories(path);
            var imgs = dirs.Sum(d => Directory.GetFiles(d, "*.*")
                .Count(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)));

            var classNames = string.Join(", ", dirs.Select(Path.GetFileName));
            _lblDatasetInfo.Text = $"Classes: {dirs.Length}  |  Images: {imgs}  |  [{classNames}]";
            _lblDatasetInfo.ForeColor = dirs.Length > 0 ? Color.Green : Color.Red;
        }
        catch (Exception ex)
        {
            _lblDatasetInfo.Text = $"Error: {ex.Message}";
            _lblDatasetInfo.ForeColor = Color.Red;
        }
    }

    private void OnStartTraining(object? sender, EventArgs e)
    {
        if (!Directory.Exists(_txtDatasetPath.Text))
        {
            MessageBox.Show("Please select a valid dataset directory.", "Warning",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        if (_btnStartTraining.Tag is "running")
            StopTraining();
        else
            _ = StartTrainingAsync();
    }

    private async Task StartTrainingAsync()
    {
        _btnStartTraining.Text = "⏹  Stop Training";
        _btnStartTraining.BackColor = Color.FromArgb(200, 0, 0);
        _btnStartTraining.Tag = "running";
        _trainingProgress.Value = 0;
        _txtTrainingLog.Clear();
        _trainingCts = new CancellationTokenSource();

        TrainingReport? report = null;

        try
        {
            var datasetPath = _txtDatasetPath.Text;
            var config = _config;
            var ct = _trainingCts.Token;

            Log("Loading dataset…");

            var (trainSamples, valSamples) = await Task.Run(() =>
                _datasetLoader.LoadAsync(datasetPath, config.ValidationSplit, ct).Result, ct);

            Log($"Train: {trainSamples.Count} samples  |  Validation: {valSamples.Count} samples");
            Log($"Starting training for {config.Epochs} epochs…");

            var progress = new Progress<EpochResult>(r =>
            {
                var pct = (int)((float)r.Epoch / config.Epochs * 100);
                _trainingProgress.Value = Math.Min(pct, 100);

                var color = r.ValidationAccuracy >= 0.9f ? Color.Cyan
                          : r.ValidationAccuracy >= 0.7f ? Color.LimeGreen
                          : Color.Yellow;

                AppendColoredLog(r.ToString(), color);
            });

            report = await Task.Run(() =>
            {
                using var trainer = BuildTrainer(config);
                return trainer.TrainAsync(trainSamples, valSamples, progress, ct).Result;
            }, ct);

            _trainingProgress.Value = 100;
            Log($"Training complete! Best accuracy: {report.BestValidationAccuracy:P2} (epoch {report.BestEpoch})");
            Log($"Model saved → {report.ModelOutputPath}");
        }
        catch (OperationCanceledException)
        {
            Log("Training cancelled.");
        }
        catch (Exception ex)
        {
            Log($"ERROR: {ex.Message}");
        }
        finally
        {
            _btnStartTraining.Text = "▶  Start Training";
            _btnStartTraining.BackColor = Color.FromArgb(0, 150, 0);
            _btnStartTraining.Tag = null;
            _trainingCts?.Dispose();
            _trainingCts = null;
        }
    }

    private void StopTraining()
    {
        _trainingCts?.Cancel();
        Log("Stop signal sent — finishing current epoch…");
    }

    // ── Logging helpers ───────────────────────────────────────────────────
    private void Log(string message)
        => AppendColoredLog(message, Color.LimeGreen);

    private void AppendColoredLog(string message, Color color)
    {
        if (InvokeRequired) { Invoke(() => AppendColoredLog(message, color)); return; }

        var ts = DateTime.Now.ToString("HH:mm:ss");
        _txtTrainingLog.SelectionStart = _txtTrainingLog.TextLength;
        _txtTrainingLog.SelectionLength = 0;
        _txtTrainingLog.AppendText($"[{ts}] {message}{Environment.NewLine}");
        _txtTrainingLog.ScrollToCaret();
    }

    // ── Factory helper ────────────────────────────────────────────────────
    private IModelTrainer BuildTrainer(MobileNetV3.Core.Configuration.TrainingConfig cfg)
    {
        var trainerLogger = NullLogger<ModelTrainer>.Instance;
        var schedulerLogger = NullLogger<LearningRateScheduler>.Instance;
        return new ModelTrainer(
            cfg,
            _datasetLoader,
            new MetricsTracker(),
            new MetricsTracker(),
            new LearningRateScheduler(cfg, schedulerLogger),
            trainerLogger);
    }
}
