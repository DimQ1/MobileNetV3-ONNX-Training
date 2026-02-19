using Size = System.Drawing.Size;

namespace MobileNetV3.UI;

public partial class MainForm
{
    // ── Shared UI factory helpers ─────────────────────────────────────────
    private static Button MakeButton(string text, Color backColor)
        => new Button
        {
            Text = text,
            BackColor = backColor,
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Font = new Font("Segoe UI", 9.5F)
        };

    private static GroupBox MakeGroupBox(string title)
        => new GroupBox
        {
            Text = title,
            Dock = DockStyle.Fill,
            Padding = new Padding(10),
            Font = new Font("Segoe UI", 9.5F)
        };
}
