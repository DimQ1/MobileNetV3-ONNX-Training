using Microsoft.Extensions.Logging;
using MobileNetV3.Core.Configuration;

namespace MobileNetV3.Core.Training;

/// <summary>
/// Планировщик скорости обучения типа ReduceLROnPlateau.
/// Уменьшает LR в <see cref="TrainingConfig.LrDecayFactor"/> раз,
/// если validation loss не улучшается на протяжении
/// <see cref="TrainingConfig.LrSchedulerPatience"/> эпох.
/// </summary>
public sealed class LearningRateScheduler
{
    private readonly TrainingConfig _config;
    private readonly ILogger<LearningRateScheduler> _logger;

    private float _currentLr;
    private float _bestLoss = float.MaxValue;
    private int _epochsWithoutImprovement;

    public float CurrentLearningRate => _currentLr;

    public LearningRateScheduler(TrainingConfig config, ILogger<LearningRateScheduler> logger)
    {
        _config = config;
        _logger = logger;
        _currentLr = config.LearningRate;
    }

    /// <summary>
    /// Вызывается после каждой эпохи. Возвращает true, если LR был уменьшен.
    /// </summary>
    public bool Step(float validationLoss)
    {
        if (validationLoss < _bestLoss - 1e-5f)
        {
            _bestLoss = validationLoss;
            _epochsWithoutImprovement = 0;
            return false;
        }

        _epochsWithoutImprovement++;

        if (_epochsWithoutImprovement >= _config.LrSchedulerPatience)
        {
            float oldLr = _currentLr;
            _currentLr *= _config.LrDecayFactor;
            _epochsWithoutImprovement = 0;

            _logger.LogInformation(
                "ReduceLROnPlateau: LR {Old:E2} → {New:E2} (нет улучшений {N} эпох)",
                oldLr, _currentLr, _config.LrSchedulerPatience);

            return true;
        }

        return false;
    }

    public void Reset()
    {
        _currentLr = _config.LearningRate;
        _bestLoss = float.MaxValue;
        _epochsWithoutImprovement = 0;
    }
}
