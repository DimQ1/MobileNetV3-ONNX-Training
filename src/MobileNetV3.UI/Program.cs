using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Preprocessing;

namespace MobileNetV3.UI;

internal static class Program
{
    [STAThread]
    static void Main()
    {
        var services = new ServiceCollection();

        services.AddLogging(builder =>
            builder.SetMinimumLevel(LogLevel.Information));

        var config = new TrainingConfig();
        services.AddSingleton(config);
        services.AddSingleton<IImagePreprocessor, ImagePreprocessor>();
        services.AddSingleton<IDatasetLoader, DatasetLoader>();

        using var serviceProvider = services.BuildServiceProvider();

        ApplicationConfiguration.Initialize();
        Application.Run(new MainForm(serviceProvider));
    }
}
