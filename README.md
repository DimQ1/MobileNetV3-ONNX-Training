# MobileNetV3 Fine-tuning — ONNX Runtime Training (C#)

Проект для **дообучения модели MobileNetV3** (классификация загрязнений линз камеры)
полностью на C# через **ONNX Runtime Training API**.

## Архитектура проекта

```
MobileNetV3Training/
├── src/
│   ├── MobileNetV3.Core/               # Библиотека ядра
│   │   ├── Abstractions/               # Интерфейсы (DI-контракты)
│   │   │   ├── IImagePreprocessor.cs
│   │   │   ├── IDatasetLoader.cs
│   │   │   ├── IModelTrainer.cs
│   │   │   └── IMetricsTracker.cs
│   │   ├── Configuration/
│   │   │   └── TrainingConfig.cs       # Все гиперпараметры и пути
│   │   ├── Data/
│   │   │   └── DatasetLoader.cs        # Загрузка + стратифицированный split
│   │   ├── Preprocessing/
│   │   │   └── ImagePreprocessor.cs    # OpenCvSharp: resize→RGB→normalize→CHW
│   │   ├── Training/
│   │   │   ├── ModelTrainer.cs         # TrainingSession, train/eval loop
│   │   │   ├── LearningRateScheduler.cs# ReduceLROnPlateau
│   │   │   └── TrainingArtifactBuilder.cs
│   │   ├── Metrics/
│   │   │   └── MetricsTracker.cs       # Накопление loss + accuracy
│   │   ├── Export/
│   │   │   └── ModelExporter.cs        # Верификация финальной модели
│   │   └── Models/
│   │       ├── ImageSample.cs          # Образец + метка
│   │       └── TrainingResult.cs       # EpochResult + TrainingReport
│   ├── MobileNetV3.Training/           # Console App (точка входа)
│   │   └── Program.cs                  # DI-setup + оркестрация
│   └── MobileNetV3.Tests/              # xUnit тесты
│       ├── Preprocessing/              # ImagePreprocessorTests
│       ├── Data/                       # DatasetLoaderTests
│       ├── Metrics/                    # MetricsTrackerTests
│       └── Training/                   # LearningRateSchedulerTests
├── scripts/
│   └── generate_training_artifacts.py  # Python: ONNX export + artifacts
├── models/                             # mobilenet_v3.onnx (после запуска скрипта)
├── artifacts/                          # training/eval/optimizer .onnx + checkpoint
├── dataset/                            # Датасет (структура папок по классам)
│   ├── Clean/
│   ├── DryDirt/
│   └── WaterDrop/
└── output/                             # Финальная обученная модель
```

## Классы для классификации

| Индекс | Метка      | Описание                         |
|--------|------------|----------------------------------|
| 0      | Clean      | Чистая линза                     |
| 1      | DryDirt    | Сухое загрязнение (пыль, грязь)  |
| 2      | WaterDrop  | Капли воды                       |

## Быстрый старт

### 1. Подготовка артефактов (Python, одноразово)

```bash
# Установка зависимостей
pip install onnxruntime-training torch torchvision onnx

# Экспорт MobileNetV3 + генерация training artifacts
python scripts/generate_training_artifacts.py
```

Скрипт создаст:
- `models/mobilenet_v3.onnx` — базовая модель (предобученная ImageNet)
- `artifacts/training_model.onnx` — граф с loss (CrossEntropyLoss)
- `artifacts/eval_model.onnx` — граф для оценки
- `artifacts/optimizer_model.onnx` — граф Adam оптимизатора
- `artifacts/checkpoint/` — начальные веса

### 2. Подготовка датасета

Поместите изображения в папки по классам:

```
dataset/
  Clean/        ← изображения чистой линзы (.jpg, .png, .bmp, ...)
  DryDirt/      ← изображения с сухим загрязнением
  WaterDrop/    ← изображения с каплями воды
```

### 3. Настройка гиперпараметров

При первом запуске создаётся `training_config.json`. Отредактируйте под свои нужды:

```json
{
  "Epochs": 15,
  "BatchSize": 16,
  "LearningRate": 0.0001,
  "ValidationSplit": 0.2,
  "UseAugmentation": true,
  "ExecutionProvider": "CPU"
}
```

### 4. Запуск дообучения

```bash
dotnet run --project src/MobileNetV3.Training
```

### 5. Запуск тестов

```bash
dotnet test src/MobileNetV3.Tests
```

## Стек технологий

| Компонент                            | Версия   | Назначение                        |
|--------------------------------------|----------|-----------------------------------|
| Microsoft.ML.OnnxRuntime.Training    | 1.20.0   | Градиентный спуск в ONNX          |
| Microsoft.ML.OnnxRuntime             | 1.20.0   | Инференс для валидации            |
| OpenCvSharp4                         | 4.10.0   | Загрузка/препроцессинг изображений|
| Microsoft.Extensions.Logging         | 9.0.5    | Структурированное логирование     |
| Microsoft.Extensions.DependencyInjection | 9.0.5 | IoC-контейнер                  |
| xUnit                                | 2.9.x    | Юнит-тесты                        |

## Препроцессинг изображений

Пайплайн обработки для каждого изображения:

```
Файл → ImRead (OpenCV BGR)
     → Resize 224×224 (Bilinear)
     → BGR → RGB
     → float32, /255.0
     → Normalize: (x - mean) / std
         mean = [0.485, 0.456, 0.406]  (ImageNet)
         std  = [0.229, 0.224, 0.225]
     → HWC → CHW layout
     → float[3 × 224 × 224]
```

### Аугментация (только train)
- Горизонтальное отражение (p=0.5)
- Случайная обрезка 90% + rescale (p=0.5)
- Яркость/контраст ±20% (p=0.5)
- Поворот ±15° (p=0.3)

## Цикл дообучения

```
для каждой эпохи:
  Train Phase:
    для каждого батча:
      TrainStep(images, labels)  → loss, gradients
      OptimizerStep()            → обновление весов
      LazyResetGrad()            → сброс градиентов

  Eval Phase:
    для каждого батча:
      EvalStep(images, labels)   → val loss
      ArgMax(logits)             → accuracy

  ReduceLROnPlateau(val_loss)
  SaveCheckpoint (если лучший результат)

Экспорт → output/mobilenet_v3_finetuned.onnx
```

## GPU ускорение

Для ускорения обучения измените `ExecutionProvider` в конфиге:

**DirectML (Windows/Xbox, любая GPU):**
```json
{ "ExecutionProvider": "DirectML", "GpuDeviceId": 0 }
```
Дополнительно установите пакет:
```bash
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```

**CUDA (NVIDIA):**
```json
{ "ExecutionProvider": "CUDA", "GpuDeviceId": 0 }
```
Дополнительно установите пакет:
```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```

## Выходная модель

После обучения файл `output/mobilenet_v3_finetuned.onnx` готов к деплою:

```csharp
// Пример инференса в production
using var session = new InferenceSession("output/mobilenet_v3_finetuned.onnx");
// Input: float[1, 3, 224, 224] NCHW
// Output: float[1, 3] — логиты для [Clean, DryDirt, WaterDrop]
```
