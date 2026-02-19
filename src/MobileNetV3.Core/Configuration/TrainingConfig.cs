using System.Text.Json.Serialization;

namespace MobileNetV3.Core.Configuration;

/// <summary>
/// Основная конфигурация процесса дообучения MobileNetV3
/// </summary>
public sealed class TrainingConfig
{
    // ─── Пути к файлам модели ────────────────────────────────────────────────

    /// <summary>Путь к базовой ONNX-модели (предобученной на ImageNet)</summary>
    public string BaseModelPath { get; set; } = "models/mobilenet_v3.onnx";

    /// <summary>Путь к ONNX-графу прямого прохода для обучения</summary>
    public string TrainingModelPath { get; set; } = "artifacts/training_model.onnx";

    /// <summary>Путь к ONNX-графу оценки (eval без dropout)</summary>
    public string EvalModelPath { get; set; } = "artifacts/eval_model.onnx";

    /// <summary>Путь к ONNX-графу оптимизатора</summary>
    public string OptimizerModelPath { get; set; } = "artifacts/optimizer_model.onnx";

    /// <summary>Директория контрольных точек (checkpoint)</summary>
    public string CheckpointDir { get; set; } = "artifacts/checkpoint";

    /// <summary>Куда сохранять финальную модель для инференса</summary>
    public string OutputModelPath { get; set; } = "output/mobilenet_v3_finetuned.onnx";

    // ─── Гиперпараметры обучения ─────────────────────────────────────────────

    /// <summary>Количество эпох дообучения</summary>
    public int Epochs { get; set; } = 15;

    /// <summary>Размер батча</summary>
    public int BatchSize { get; set; } = 16;

    /// <summary>Начальная скорость обучения для Adam</summary>
    public float LearningRate { get; set; } = 1e-4f;

    /// <summary>L2-регуляризация (weight decay)</summary>
    public float WeightDecay { get; set; } = 1e-5f;

    /// <summary>Beta1 для Adam</summary>
    public float AdamBeta1 { get; set; } = 0.9f;

    /// <summary>Beta2 для Adam</summary>
    public float AdamBeta2 { get; set; } = 0.999f;

    /// <summary>Epsilon для Adam (предотвращение деления на 0)</summary>
    public float AdamEpsilon { get; set; } = 1e-8f;

    /// <summary>Коэффициент уменьшения LR по расписанию (ReduceOnPlateau)</summary>
    public float LrDecayFactor { get; set; } = 0.5f;

    /// <summary>Терпение планировщика LR (эпох без улучшения)</summary>
    public int LrSchedulerPatience { get; set; } = 3;

    // ─── Параметры данных и препроцессинга ───────────────────────────────────

    /// <summary>Корневая директория датасета</summary>
    public string DatasetRootDir { get; set; } = "dataset";

    /// <summary>Целевой размер изображения (224 для MobileNetV3)</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>Доля выборки для валидации [0..1]</summary>
    public float ValidationSplit { get; set; } = 0.2f;

    /// <summary>Средние значения каналов ImageNet для нормализации (RGB)</summary>
    public float[] NormMean { get; set; } = [0.485f, 0.456f, 0.406f];

    /// <summary>Стандартные отклонения каналов ImageNet для нормализации (RGB)</summary>
    public float[] NormStd { get; set; } = [0.229f, 0.224f, 0.225f];

    /// <summary>Применять аугментацию к обучающей выборке</summary>
    public bool UseAugmentation { get; set; } = true;

    // ─── Классы ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Метки классов. По умолчанию: загрязнение линз камеры.
    /// Порядок соответствует выходному индексу модели.
    /// </summary>
    public string[] ClassLabels { get; set; } = ["Clean", "DryDirt", "WaterDrop"];

    /// <summary>Количество выходных классов (вычисляется автоматически)</summary>
    [JsonIgnore]
    public int NumClasses => ClassLabels.Length;

    // ─── Имена узлов ONNX-графа ──────────────────────────────────────────────

    /// <summary>Имя входного тензора (изображение)</summary>
    public string InputNodeName { get; set; } = "input";

    /// <summary>Имя выходного тензора (логиты)</summary>
    public string OutputNodeName { get; set; } = "output";

    /// <summary>Имя входного тензора меток (для вычисления loss)</summary>
    public string LabelNodeName { get; set; } = "labels";

    // ─── Система и ускорение ─────────────────────────────────────────────────

    /// <summary>Провайдер выполнения: CPU, DirectML, CUDA</summary>
    public ExecutionProvider ExecutionProvider { get; set; } = ExecutionProvider.CPU;

    /// <summary>Индекс GPU (для DirectML/CUDA)</summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>Сохранять checkpoint каждые N эпох</summary>
    public int SaveCheckpointEveryNEpochs { get; set; } = 5;

    /// <summary>Логировать метрики каждые N батчей</summary>
    public int LogEveryNBatches { get; set; } = 10;
}

/// <summary>Провайдер выполнения ONNX Runtime</summary>
public enum ExecutionProvider
{
    CPU,
    DirectML,
    CUDA
}
