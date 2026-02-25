using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MobileNetV3.Core.Export;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OcvSize = OpenCvSharp.Size;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace MobileNetV3.UI;

public partial class MainForm
{
    private void BuildImageTestTab()
    {
        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical
        };
        split.Panel1.Controls.Add(BuildImagePreviewPanel());
        split.Panel2.Controls.Add(BuildImageResultsPanel());
        _imageTestTab.Controls.Add(split);

        // Set splitter properties when form is shown
        this.Shown += (s, e) =>
        {
            try
            {
                split.Panel1MinSize = 400;
                split.Panel2MinSize = 280;
                
                int availableWidth = split.Width - split.Panel2MinSize;
                int desiredDistance = Math.Min(680, Math.Max(400, availableWidth));
                split.SplitterDistance = desiredDistance;
            }
            catch { /* Ignore if splitter configuration fails */ }
        };
    }

    private Panel BuildImagePreviewPanel()
    {
        var root = new Panel { Dock = DockStyle.Fill, Padding = new Padding(12) };
        var grp = MakeGroupBox("Image Preview");

        _imagePreview = new PictureBox
        {
            Dock = DockStyle.Fill,
            SizeMode = PictureBoxSizeMode.Zoom,
            BackColor = Color.FromArgb(40, 40, 40),
            BorderStyle = BorderStyle.FixedSingle
        };

        var bar = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 48,
            FlowDirection = FlowDirection.LeftToRight,
            Padding = new Padding(0, 4, 0, 4)
        };

        _btnSelectImage = MakeButton("Open Image", Color.FromArgb(0, 120, 215));
        _btnPredictImage = MakeButton("Predict", Color.FromArgb(0, 150, 0));
        _btnSelectImage.Size = new Size(130, 34);
        _btnPredictImage.Size = new Size(110, 34);
        _btnSelectImage.Margin = new Padding(0, 0, 8, 0);
        _btnPredictImage.Margin = new Padding(0, 0, 8, 0);

        _btnSelectImage.Click += OnSelectImage;
        _btnPredictImage.Click += OnPredictImage;

        bar.Controls.AddRange(new Control[] { _btnSelectImage, _btnPredictImage });
        grp.Controls.Add(_imagePreview);
        grp.Controls.Add(bar);
        root.Controls.Add(grp);
        return root;
    }

    private Panel BuildImageResultsPanel()
    {
        var root = new Panel { Dock = DockStyle.Fill, Padding = new Padding(12) };
        var grp = MakeGroupBox("Prediction Results");

        _lblImagePrediction = new Label
        {
            Text = "No prediction yet",
            Font = new Font("Segoe UI", 13F, FontStyle.Bold),
            AutoSize = true,
            Location = new Point(10, 24),
            ForeColor = Color.White
        };

        _lstImageResults = new ListBox
        {
            Font = new Font("Consolas", 10F),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.LimeGreen,
            Location = new Point(10, 60),
            Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right
        };

        grp.Controls.AddRange(new Control[] { _lblImagePrediction, _lstImageResults });
        grp.Resize += (_, _) =>
            _lstImageResults.Size = new Size(grp.ClientSize.Width - 30, grp.ClientSize.Height - 80);

        root.Controls.Add(grp);
        return root;
    }

    private void OnSelectImage(object? sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Title = "Select image",
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*"
        };
        if (dlg.ShowDialog() != DialogResult.OK) return;

        _currentImagePath = dlg.FileName;
        _imagePreview.Image?.Dispose();
        _imagePreview.Image = Image.FromFile(_currentImagePath);
        _lstImageResults.Items.Clear();
        _lblImagePrediction.Text = "Ready - press Predict";
        _lblImagePrediction.ForeColor = Color.White;
    }

    private async void OnPredictImage(object? sender, EventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_currentImagePath))
        {
            MessageBox.Show("Please open an image first.", "No image",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        _btnPredictImage.Enabled = false;
        _lblImagePrediction.Text = "Running...";
        _lblImagePrediction.ForeColor = Color.Yellow;

        try
        {
            var (top, scores) = await Task.Run(() => RunImageInference(_currentImagePath!));

            _lstImageResults.Items.Clear();
            foreach (var (cls, conf) in scores.OrderByDescending(x => x.Conf))
                _lstImageResults.Items.Add(string.Format("{0,-18} {1,6:F2} %", cls, conf * 100));

            _lblImagePrediction.Text = string.Format("{0}  ({1:F1} %)", top.Class, top.Conf * 100);
            _lblImagePrediction.ForeColor = top.Conf >= 0.70f ? Color.LimeGreen : Color.Orange;
        }
        catch (FileNotFoundException)
        {
            _lblImagePrediction.Text = "Model not found - train first!";
            _lblImagePrediction.ForeColor = Color.OrangeRed;
        }
        catch (Exception ex)
        {
            _lblImagePrediction.Text = "Error: " + ex.Message;
            _lblImagePrediction.ForeColor = Color.Red;
        }
        finally
        {
            _btnPredictImage.Enabled = true;
        }
    }

    private ((string Class, float Conf) Top, List<(string Class, float Conf)> All)
        RunImageInference(string imagePath)
    {
        var modelPath = Path.Combine("models", "mobilenet_v3.onnx");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model not found", modelPath);

        var metaPath = Path.Combine("models", "mobilenet_v3.json");
        var meta = ModelInfo.Load(metaPath);

        using var mat = Cv2.ImRead(imagePath);
        if (mat.Empty()) throw new Exception("Cannot read image file.");

        var floatData = _preprocessor.PreprocessMat(mat);
        var tensor = FloatArrayToDenseTensor(floatData);
        var scores = RunOnnxSession(modelPath, tensor, meta.ClassNames);
        var top = scores.OrderByDescending(x => x.Conf).First();
        return (top, scores);
    }

    private static List<(string Class, float Conf)> RunOnnxSession(
        string modelPath, DenseTensor<float> tensor, string[] classNames)
    {
        using var session = new InferenceSession(modelPath);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor<float>("input", tensor) };
        using var results = session.Run(inputs);

        var raw = results.First().AsTensor<float>().ToArray();
        var softmax = Softmax(raw);
        return classNames.Zip(softmax, (c, s) => (c, s)).ToList();
    }

    private static DenseTensor<float> FloatArrayToDenseTensor(float[] data)
    {
        // shape: [1, 3, 224, 224]
        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        data.CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    private static float[] Softmax(float[] x)
    {
        var max = x.Max();
        var exp = x.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }
}
