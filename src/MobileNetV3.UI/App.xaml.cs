using System.Windows;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Data;
using MobileNetV3.Core.Preprocessing;

namespace MobileNetV3.UI;

public partial class App : Application
{
    public IServiceProvider ServiceProvider { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        var services = new ServiceCollection();

        services.AddLogging(builder =>
            builder.SetMinimumLevel(LogLevel.Information));

        var config = new TrainingConfig();
        services.AddSingleton(config);
        services.AddSingleton<IImagePreprocessor, ImagePreprocessor>();
        services.AddSingleton<IDatasetLoader, DatasetLoader>();

        ServiceProvider = services.BuildServiceProvider();

        base.OnStartup(e);
    }
}
