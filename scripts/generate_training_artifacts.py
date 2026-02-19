"""
Скрипт генерации артефактов обучения для ONNX Runtime Training.

Запуск:
    pip install onnxruntime-training torch torchvision onnx
    python scripts/generate_training_artifacts.py

Результат:
    artifacts/training_model.onnx   — граф прямого прохода с loss
    artifacts/eval_model.onnx       — граф оценки (без dropout)
    artifacts/optimizer_model.onnx  — граф оптимизатора Adam
    artifacts/checkpoint/           — начальные веса модели
"""

import os
import urllib.request
import torch
import torchvision.models as tv_models
import onnx
from onnxruntime.training import artifacts


# ─── Конфигурация ─────────────────────────────────────────────────────────────

CONFIG = {
    "num_classes": 3,
    "class_labels": ["Clean", "DryDirt", "WaterDrop"],
    "image_size": 224,
    "base_model_path": "models/mobilenet_v3.onnx",
    "artifacts_dir": "artifacts",
    "checkpoint_dir": "artifacts/checkpoint",
    # Узлы, веса которых будут обучаться (last classifier + новый слой)
    "trainable_params": [
        "classifier.3.weight",
        "classifier.3.bias",
        "classifier.0.weight",
        "classifier.0.bias",
    ],
}


def export_base_model(num_classes: int, output_path: str) -> None:
    """
    Загружает предобученный MobileNetV3-Small, заменяет финальный слой
    под нужное количество классов и экспортирует в ONNX.
    """
    print(f"[1/4] Загрузка MobileNetV3-Small (pretrained=ImageNet)...")

    model = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Заменяем classifier под наши классы
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)

    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[2/4] Экспорт базовой модели → {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        verbose=False,
    )
    print(f"    Базовая модель сохранена: {output_path}")


def generate_artifacts(base_model_path: str, artifacts_dir: str,
                        checkpoint_dir: str, trainable_params: list[str]) -> None:
    """
    Генерирует артефакты ONNX Runtime Training через официальный API.
    """
    print(f"[3/4] Генерация артефактов обучения в {artifacts_dir}...")

    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    onnx_model = onnx.load(base_model_path)

    # Получаем имена всех обучаемых параметров из модели
    all_params = [init.name for init in onnx_model.graph.initializer]

    # Если trainable_params пустой — обучаем все (full fine-tuning)
    # Иначе — только указанные (layer freezing)
    requires_grad = [
        name for name in all_params
        if any(tp in name for tp in trainable_params)
    ] if trainable_params else all_params

    print(f"    Обучаемые параметры ({len(requires_grad)} из {len(all_params)}):")
    for p in requires_grad:
        print(f"      - {p}")

    artifacts.generate_artifacts(
        onnx_model,
        requires_grad=requires_grad,
        frozen_params=[p for p in all_params if p not in requires_grad],
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=artifacts_dir,
    )

    # Сохраняем checkpoint с начальными весами
    print(f"    Checkpoint сохранён в: {checkpoint_dir}")


def verify_artifacts(artifacts_dir: str, checkpoint_dir: str) -> None:
    """Проверяет наличие всех необходимых файлов."""
    print(f"[4/4] Верификация артефактов...")

    required = [
        os.path.join(artifacts_dir, "training_model.onnx"),
        os.path.join(artifacts_dir, "eval_model.onnx"),
        os.path.join(artifacts_dir, "optimizer_model.onnx"),
    ]

    for path in required:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ ОТСУТСТВУЕТ"
        size = f"({os.path.getsize(path) // 1024} KB)" if exists else ""
        print(f"    {status} {path} {size}")

    print("\nАртефакты готовы! Теперь запустите C#-приложение:")
    print("    dotnet run --project src/MobileNetV3.Training")


if __name__ == "__main__":
    print("=" * 60)
    print("  MobileNetV3 Training Artifacts Generator")
    print("=" * 60)

    # Шаг 1-2: Экспорт базовой модели
    export_base_model(
        num_classes=CONFIG["num_classes"],
        output_path=CONFIG["base_model_path"],
    )

    # Шаг 3: Генерация артефактов обучения
    generate_artifacts(
        base_model_path=CONFIG["base_model_path"],
        artifacts_dir=CONFIG["artifacts_dir"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
        trainable_params=CONFIG["trainable_params"],
    )

    # Шаг 4: Верификация
    verify_artifacts(
        artifacts_dir=CONFIG["artifacts_dir"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
    )

    print("\n" + "=" * 60)
    print("  Готово!")
    print("=" * 60)
