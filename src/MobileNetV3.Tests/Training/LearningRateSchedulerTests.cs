using Microsoft.Extensions.Logging.Abstractions;
using MobileNetV3.Core.Configuration;
using MobileNetV3.Core.Training;

namespace MobileNetV3.Tests.Training;

public sealed class LearningRateSchedulerTests
{
    private static LearningRateScheduler CreateScheduler(
        float lr = 1e-3f,
        float decay = 0.5f,
        int patience = 3)
    {
        var config = new TrainingConfig
        {
            LearningRate = lr,
            LrDecayFactor = decay,
            LrSchedulerPatience = patience
        };
        return new LearningRateScheduler(config, NullLogger<LearningRateScheduler>.Instance);
    }

    [Fact]
    public void InitialLR_EqualsConfigValue()
    {
        var scheduler = CreateScheduler(lr: 0.001f);

        Assert.Equal(0.001f, scheduler.CurrentLearningRate, precision: 6);
    }

    [Fact]
    public void Step_ImprovingLoss_DoesNotReduceLR()
    {
        var scheduler = CreateScheduler(patience: 3);

        bool reduced = scheduler.Step(1.0f);
        Assert.False(reduced);
        Assert.Equal(1e-3f, scheduler.CurrentLearningRate, precision: 6);

        reduced = scheduler.Step(0.8f);  // улучшение
        Assert.False(reduced);
    }

    [Fact]
    public void Step_AfterPatience_ReducesLR()
    {
        var scheduler = CreateScheduler(lr: 1e-3f, decay: 0.5f, patience: 3);

        scheduler.Step(1.0f); // epoch 1 — лучшая
        scheduler.Step(1.0f); // epoch 2 — нет улучшения (1)
        scheduler.Step(1.0f); // epoch 3 — нет улучшения (2)
        bool reduced = scheduler.Step(1.0f); // epoch 4 — нет улучшения (3) → reduce

        Assert.True(reduced);
        Assert.Equal(5e-4f, scheduler.CurrentLearningRate, precision: 6);
    }

    [Fact]
    public void Step_AfterReduction_ResetsCounter()
    {
        var scheduler = CreateScheduler(lr: 1e-3f, decay: 0.1f, patience: 2);

        scheduler.Step(1.0f);
        scheduler.Step(1.0f);
        scheduler.Step(1.0f); // → reduce (counter = 2)

        // После reduce счётчик обнуляется, следующие шаги не должны сразу снижать
        bool reduced = scheduler.Step(1.0f); // 1 без улучшения (не patience=2 ещё)
        Assert.False(reduced);
    }

    [Fact]
    public void Step_ConsecutiveReductions_MultipliesDecay()
    {
        // patience=2: для reduce нужно 2 эпохи без улучшения
        // Шаг 1: best=1.0
        // Шаг 2: counter=1
        // Шаг 3: counter=2 → reduce → lr=5e-3, counter=0
        // Шаг 4: counter=1
        // Шаг 5: counter=2 → reduce → lr=2.5e-3, counter=0
        var scheduler = CreateScheduler(lr: 1e-2f, decay: 0.5f, patience: 2);

        scheduler.Step(1.0f); // best = 1.0
        scheduler.Step(1.0f); // counter = 1
        scheduler.Step(1.0f); // counter = 2 → 1й reduce → lr = 5e-3
        scheduler.Step(1.0f); // counter = 1
        scheduler.Step(1.0f); // counter = 2 → 2й reduce → lr = 2.5e-3

        Assert.Equal(2.5e-3f, scheduler.CurrentLearningRate, precision: 5);
    }

    [Fact]
    public void Step_ImprovementResetsPatience()
    {
        var scheduler = CreateScheduler(lr: 1e-3f, decay: 0.5f, patience: 3);

        scheduler.Step(1.0f); // best = 1.0
        scheduler.Step(1.0f); // 1 без улучшения
        scheduler.Step(1.0f); // 2 без улучшения
        scheduler.Step(0.5f); // улучшение → сброс счётчика
        scheduler.Step(0.5f); // 1 без улучшения
        scheduler.Step(0.5f); // 2 без улучшения
        bool reduced = scheduler.Step(0.5f); // 3 без улучшения → reduce

        Assert.True(reduced);
    }

    [Fact]
    public void Reset_RestoresInitialLR()
    {
        var scheduler = CreateScheduler(lr: 1e-3f, decay: 0.5f, patience: 1);

        scheduler.Step(1.0f);
        scheduler.Step(1.0f); // reduce → lr = 5e-4

        scheduler.Reset();

        Assert.Equal(1e-3f, scheduler.CurrentLearningRate, precision: 6);
    }

    [Fact]
    public void Step_SlightImprovementBelowThreshold_TreatedAsNoImprovement()
    {
        var scheduler = CreateScheduler(patience: 2);

        scheduler.Step(1.0f);      // best = 1.0
        scheduler.Step(0.99999f);  // delta < 1e-5 — считается "нет улучшения"
        bool reduced = scheduler.Step(0.99999f); // 2 без улучшения → reduce

        Assert.True(reduced);
    }
}
