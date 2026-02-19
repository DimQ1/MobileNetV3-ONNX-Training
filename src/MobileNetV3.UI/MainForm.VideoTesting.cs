using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Export;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OcvPoint = OpenCvSharp.Point;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace MobileNetV3.UI;

public partial class MainForm
{
    private void BuildVideoTestTab()
    {
        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterDistance = 820,
            Panel1MinSize = 500,
            Panel2MinSize = 240
        };
        split.Panel1.Controls.Add(BuildVideoPreviewPanel());
        split.Panel2.Controls.Add(BuildVideoResultsPanel());
        _videoTestTab.Controls.Add(split);
    }

    private Panel BuildVideoPreviewPanel()
    {
        var root = new Panel { Dock = DockStyle.Fill, Padding = new Padding(12) };
        var grp = MakeGroupBox("Video / Webcam Preview");

        _videoPreview = new PictureBox
        {
            Dock = DockStyle.Fill,
            SizeMode = PictureBoxSizeMode.Zoom,
            BackColor = Color.Black,
            BorderStyle = BorderStyle.FixedSingle
        };

        var bar = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 54,
            FlowDirection = FlowDirection.LeftToRight,
            Padding = new Padding(0, 6, 0, 0)
        };

        _cmbVideoSource = new ComboBox
        {
            Size = new Size(160, 32),
            DropDownStyle = ComboBoxStyle.DropDownList,
            Margin = new Padding(0, 0, 6, 0)
        };
        _cmbVideoSource.Items.AddRange(new object[] { "Camera 0", "Camera 1", "Camera 2" });
        _cmbVideoSource.SelectedIndex = 0;

        _btnSelectVideo = MakeButton("Open Video", Color.FromArgb(0, 120, 215));
        _btnStartWebcam = MakeButton("Start Webcam", Color.FromArgb(0, 140, 0));
        _btnStartVideo = MakeButton("Start Video", Color.FromArgb(0, 140, 0));
        _btnStopVideo = MakeButton("Stop", Color.FromArgb(200, 0, 0));

        foreach (var b in new[] { _btnSelectVideo, _btnStartWebcam, _btnStartVideo, _btnStopVideo })
        {
            b.Size = new Size(120, 34);
            b.Margin = new Padding(0, 0, 6, 0);
        }
        _btnStopVideo.Enabled = false;

        _btnSelectVideo.Click += OnSelectVideo;
        _btnStartWebcam.Click += OnStartWebcam;
        _btnStartVideo.Click += OnStartVideo;
        _btnStopVideo.Click += OnStopVideo;

        bar.Controls.AddRange(new Control[]
            { _cmbVideoSource, _btnSelectVideo, _btnStartWebcam, _btnStartVideo, _btnStopVideo });

        _lblVideoStatus = new Label
        {
            Text = "Status: Idle",
            Dock = DockStyle.Bottom,
            Height = 24,
            ForeColor = Color.Gray,
            Font = new Font("Segoe UI", 9F)
        };

        grp.Controls.Add(_videoPreview);
        grp.Controls.Add(bar);
        grp.Controls.Add(_lblVideoStatus);
        root.Controls.Add(grp);
        return root;
    }

    private Panel BuildVideoResultsPanel()
    {
        var root = new Panel { Dock = DockStyle.Fill, Padding = new Padding(12) };
        var grp = MakeGroupBox("Live Predictions");

        _lstVideoResults = new ListBox
        {
            Dock = DockStyle.Fill,
            Font = new Font("Consolas", 9.5F),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.LimeGreen
        };

        grp.Controls.Add(_lstVideoResults);
        root.Controls.Add(grp);
        return root;
    }

    private void OnSelectVideo(object? sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Title = "Select video file",
            Filter = "Video files|*.mp4;*.avi;*.mov;*.wmv;*.mkv|All files|*.*"
        };
        if (dlg.ShowDialog() != DialogResult.OK) return;

        _currentVideoPath = dlg.FileName;
        _cmbVideoSource.SelectedIndex = -1;
        _lblVideoStatus.Text = "Loaded: " + Path.GetFileName(_currentVideoPath);
        _lblVideoStatus.ForeColor = Color.CornflowerBlue;
    }

    private async void OnStartWebcam(object? sender, EventArgs e)
    {
        int camId = Math.Max(_cmbVideoSource.SelectedIndex, 0);
        await RunStreamAsync(camId.ToString(), isWebcam: true);
    }

    private async void OnStartVideo(object? sender, EventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_currentVideoPath))
        {
            MessageBox.Show("Please select a video file first.", "No video",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }
        await RunStreamAsync(_currentVideoPath!, isWebcam: false);
    }

    private void OnStopVideo(object? sender, EventArgs e)
    {
        _videoCts?.Cancel();
        _lblVideoStatus.Text = "Status: Stopping...";
        _lblVideoStatus.ForeColor = Color.Orange;
    }

    private async Task RunStreamAsync(string source, bool isWebcam)
    {

        _btnStartVideo.Enabled = false;
        _btnStartWebcam.Enabled = false;
        _btnStopVideo.Enabled = true;
        _videoCts = new CancellationTokenSource();

        _lblVideoStatus.Text = isWebcam ? "Status: Webcam running" : "Status: Video playing";
        _lblVideoStatus.ForeColor = Color.LimeGreen;

        try
        {
            var ct = _videoCts.Token;
            await Task.Run(() => ProcessStream(source, isWebcam, ct), ct);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Invoke(() => MessageBox.Show("Stream error: " + ex.Message, "Error",
                MessageBoxButtons.OK, MessageBoxIcon.Error));
        }
        finally
        {

            _videoCts?.Dispose();
            _videoCts = null;
            Invoke(() =>
            {
                _btnStartVideo.Enabled = true;
                _btnStartWebcam.Enabled = true;
                _btnStopVideo.Enabled = false;
                _lblVideoStatus.Text = "Status: Idle";
                _lblVideoStatus.ForeColor = Color.Gray;
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

                // Run inference every 500 ms
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

                    Invoke(() =>
                    {
                        _lstVideoResults.Items.Insert(0,
                            string.Format("[{0}] {1}  {2:F1}%",
                                DateTime.Now.ToString("HH:mm:ss"), lastLabel, lastConf * 100));
                        if (_lstVideoResults.Items.Count > 60)
                            _lstVideoResults.Items.RemoveAt(_lstVideoResults.Items.Count - 1);
                    });
                }

                // Draw overlay on frame
                DrawOverlay(frame, lastLabel, lastConf, frameIdx, modelReady);

                var bmp = BitmapConverter.ToBitmap(frame);
                Invoke(() =>
                {
                    var old = _videoPreview.Image;
                    _videoPreview.Image = bmp;
                    old?.Dispose();
                });

                // Throttle to ~30 fps
                Thread.Sleep(33);
            }
        }
    }

    private static void DrawOverlay(Mat frame, string label, float conf, int frameIdx, bool modelReady)
    {
        var fnt = HersheyFonts.HersheySimplex;
        var scale = 0.9;
        var thick = 2;

        // Frame counter
        Cv2.PutText(frame, "Frame: " + frameIdx,
            new OcvPoint(10, frame.Rows - 10), fnt, 0.55,
            new Scalar(180, 180, 180), 1);

        if (!modelReady)
        {
            Cv2.PutText(frame, "No model - train first",
                new OcvPoint(10, 35), fnt, scale,
                new Scalar(0, 100, 255), thick);
            return;
        }

        var text = string.Format("{0}  {1:F1}%", label, conf * 100);
        var color = conf >= 0.70 ? new Scalar(0, 230, 0) : new Scalar(0, 165, 255);
        var txtSize = Cv2.GetTextSize(text, fnt, scale, thick, out var baseline);

        // Background rect
        Cv2.Rectangle(frame,
            new OcvPoint(6, 6),
            new OcvPoint(txtSize.Width + 14, txtSize.Height + baseline + 14),
            new Scalar(0, 0, 0, 160), -1);

        Cv2.PutText(frame, text, new OcvPoint(10, txtSize.Height + 10),
            fnt, scale, color, thick);
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        _trainingCts?.Cancel();
        _videoCts?.Cancel();
        _imagePreview.Image?.Dispose();
        _videoPreview.Image?.Dispose();
        base.OnFormClosing(e);
    }
}
