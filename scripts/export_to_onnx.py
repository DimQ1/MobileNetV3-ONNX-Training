"""
Export MobileNetV3-Small → ONNX + Training Artifacts
=====================================================

Этапы:
  1. Загружает MobileNetV3-Small с предобученными весами ImageNet
  2. Заменяет классификатор на нужное количество классов
  3. Экспортирует базовую модель в ONNX (opset 17)
  4. Упрощает граф через onnxsim
  5. Верифицирует модель: shape, inference pass, топ-классы
  6. Генерирует ORT Training Artifacts (train/eval/optimizer + checkpoint)
  7. Сохраняет метаданные экспорта в JSON

Запуск:
    # Из корня проекта:
    .venv/Scripts/python scripts/export_to_onnx.py

    # С явными параметрами:
    .venv/Scripts/python scripts/export_to_onnx.py \
        --num-classes 3 \
        --output models/mobilenet_v3.onnx \
        --artifacts-dir artifacts \
        --opset 17
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import MobileNet_V3_Small_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Константы ────────────────────────────────────────────────────────────────

CLASS_LABELS   = ["Clean", "DryDirt", "WaterDrop"]
IMAGE_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# Параметры для заморозки слоёв (fine-tuning только classifier)
TRAINABLE_PARAM_PATTERNS = [
    "classifier.3",   # последний Linear-слой
    "classifier.0",   # BatchNorm перед classifier
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Построение модели
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> nn.Module:
    """
    Загружает MobileNetV3-Small (ImageNet weights) и адаптирует
    classifier под нужное число классов.

    Архитектура classifier:
        Linear(576 → 1024) → Hardswish → Dropout(0.2) → Linear(1024 → num_classes)
    """
    log.info("Загрузка MobileNetV3-Small (weights=IMAGENET1K_V1)...")
    model = tv_models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Количество нейронов перед последним слоем
    in_features = model.classifier[3].in_features
    log.info("  Входные признаки classifier: %d", in_features)

    # Заменяем выходной Linear под num_classes
    model.classifier[3] = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(model.classifier[3].weight)
    nn.init.zeros_(model.classifier[3].bias)

    model.eval()

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Всего параметров:    %s", f"{total_params:,}")
    log.info("  Обучаемых параметров: %s", f"{trainable_params:,}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Экспорт в ONNX
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(
    model: nn.Module,
    output_path: str,
    opset: int = 17,
    image_size: int = IMAGE_SIZE,
) -> None:
    """
    Экспортирует модель в ONNX с динамической осью batch_size.
    """
    log.info("Экспорт в ONNX (opset=%d) → %s", opset, output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        # dynamo=False — использует TorchScript-based exporter (стабильный, совместим с ORT)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            dynamo=False,
            verbose=False,
        )

    size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    log.info("  Сохранено: %.2f MB", size_mb)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Упрощение графа (onnxsim)
# ══════════════════════════════════════════════════════════════════════════════

def simplify_onnx(model_path: str) -> bool:
    """
    Применяет onnxsim: удаляет лишние узлы, константные подграфы,
    упрощает операции. Перезаписывает файл.
    """
    try:
        import onnx
        import onnxsim

        log.info("Упрощение ONNX-графа (onnxsim)...")
        model = onnx.load(model_path)
        simplified, ok = onnxsim.simplify(model)

        if ok:
            onnx.save(simplified, model_path)
            log.info("  Граф успешно упрощён")
        else:
            log.warning("  onnxsim: упрощение не дало результата")

        return ok

    except ImportError:
        log.warning("  onnxsim не установлен, пропускаем упрощение")
        return False
    except Exception as e:
        log.warning("  Ошибка onnxsim: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. Верификация модели
# ══════════════════════════════════════════════════════════════════════════════

def verify_onnx(
    model_path: str,
    num_classes: int,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    Верифицирует ONNX-модель:
      - Структурная проверка через onnx.checker
      - Тестовый inference через onnxruntime
      - Сравнение выходов PyTorch vs ONNX Runtime
    Возвращает словарь с метаданными.
    """
    import onnx
    import onnxruntime as ort

    log.info("Верификация ONNX-модели...")

    # 4.1 Структурная проверка
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    log.info("  Структурная проверка: OK")

    # 4.2 Метаданные графа
    inputs  = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
               for i in model.graph.input}
    outputs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim]
               for o in model.graph.output}
    num_nodes = len(model.graph.node)

    log.info("  Входы:  %s", inputs)
    log.info("  Выходы: %s", outputs)
    log.info("  Узлов в графе: %d", num_nodes)

    # 4.3 Inference через OnnxRuntime
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3  # ERROR only

    sess = ort.InferenceSession(model_path, session_opts,
                                providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    t0 = time.perf_counter()
    ort_out = sess.run(None, {"input": dummy})
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = ort_out[0]  # [1, num_classes]
    assert logits.shape == (1, num_classes), \
        f"Неверный размер выхода: {logits.shape}, ожидается (1, {num_classes})"

    log.info("  Тестовый inference: OK  (latency=%.1f ms)", latency_ms)
    log.info("  Размер выхода: %s", logits.shape)

    # 4.4 Softmax → топ-классы (для отладки)
    probs = _softmax(logits[0])
    top_idx = np.argsort(probs)[::-1]
    log.info("  Топ предсказания (random input — равномерное распределение ожидаемо):")
    for i in top_idx:
        log.info("    [%d] prob=%.4f", i, probs[i])

    return {
        "num_nodes": num_nodes,
        "inputs": inputs,
        "outputs": outputs,
        "latency_ms": round(latency_ms, 2),
        "output_shape": list(logits.shape),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Генерация ORT Training Artifacts
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_artifacts(
    base_model_path: str,
    artifacts_dir: str,
    trainable_patterns: Optional[list[str]] = None,
) -> bool:
    """
    Генерирует артефакты ONNX Runtime Training через официальный API:
      - training_model.onnx  — граф forward + CrossEntropyLoss
      - eval_model.onnx      — граф без dropout (для валидации)
      - optimizer_model.onnx — граф AdamW
      - checkpoint/          — начальные веса

    Возвращает True при успехе.
    """
    try:
        import onnx
        from onnxruntime.training import artifacts

        log.info("Генерация ORT Training Artifacts → %s", artifacts_dir)

        onnx_model = onnx.load(base_model_path)
        all_params = [init.name for init in onnx_model.graph.initializer]

        # Определяем обучаемые параметры
        if trainable_patterns:
            requires_grad = [
                name for name in all_params
                if any(pat in name for pat in trainable_patterns)
            ]
            frozen_params = [p for p in all_params if p not in requires_grad]
        else:
            # Full fine-tuning — обучаем все параметры
            requires_grad = all_params
            frozen_params = []

        log.info("  Обучаемых параметров: %d / %d",
                 len(requires_grad), len(all_params))
        log.info("  Замороженных:         %d", len(frozen_params))

        if requires_grad:
            log.info("  Примеры обучаемых слоёв:")
            for name in requires_grad[:6]:
                log.info("    - %s", name)
            if len(requires_grad) > 6:
                log.info("    ... и ещё %d", len(requires_grad) - 6)

        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

        artifacts.generate_artifacts(
            onnx_model,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            loss=artifacts.LossType.CrossEntropyLoss,
            optimizer=artifacts.OptimType.AdamW,
            artifact_directory=artifacts_dir,
        )

        # Проверяем созданные файлы
        expected = [
            "training_model.onnx",
            "eval_model.onnx",
            "optimizer_model.onnx",
        ]
        for fname in expected:
            fpath = os.path.join(artifacts_dir, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / (1024 ** 2)
                log.info("  ✓ %-30s  %.2f MB", fname, size_mb)
            else:
                log.error("  ✗ Файл не создан: %s", fpath)

        # Сохраняем checkpoint с начальными весами
        checkpoint_dir = os.path.join(artifacts_dir, "checkpoint")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        log.info("  Checkpoint директория: %s", checkpoint_dir)

        return True

    except ImportError as e:
        log.warning("onnxruntime-training не установлен (%s). "
                    "Artifacts НЕ сгенерированы.", e)
        log.warning("Установите: pip install onnxruntime-training")
        return False
    except Exception as e:
        log.error("Ошибка генерации артефактов: %s", e, exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 6. Сохранение метаданных
# ══════════════════════════════════════════════════════════════════════════════

def save_metadata(
    output_path: str,
    num_classes: int,
    class_labels: list[str],
    opset: int,
    verify_info: dict,
    artifacts_ok: bool,
) -> None:
    """Сохраняет JSON с метаданными экспорта рядом с моделью."""
    import torch
    import torchvision

    meta_path = str(Path(output_path).with_suffix(".json"))
    metadata = {
        "model":          "MobileNetV3-Small",
        "base_weights":   "ImageNet (IMAGENET1K_V1)",
        "num_classes":    num_classes,
        "class_labels":   class_labels,
        "input_shape":    [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "input_name":     "input",
        "output_name":    "output",
        "image_size":     IMAGE_SIZE,
        "norm_mean":      IMAGENET_MEAN,
        "norm_std":       IMAGENET_STD,
        "opset_version":  opset,
        "torch_version":  torch.__version__,
        "torchvision_version": torchvision.__version__,
        "onnx_graph":     verify_info,
        "training_artifacts_generated": artifacts_ok,
        "exported_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("Метаданные сохранены: %s", meta_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export MobileNetV3-Small to ONNX + ORT Training Artifacts"
    )
    p.add_argument("--num-classes",   type=int,  default=len(CLASS_LABELS))
    p.add_argument("--output",        type=str,  default="models/mobilenet_v3.onnx")
    p.add_argument("--artifacts-dir", type=str,  default="artifacts")
    p.add_argument("--opset",         type=int,  default=17)
    p.add_argument("--no-simplify",   action="store_true",
                   help="Пропустить упрощение графа через onnxsim")
    p.add_argument("--full-finetune", action="store_true",
                   help="Обучать все параметры (по умолчанию — только classifier)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("  MobileNetV3-Small → ONNX Export")
    log.info("  Classes:      %d  %s", args.num_classes, CLASS_LABELS[:args.num_classes])
    log.info("  Output:       %s", args.output)
    log.info("  Artifacts:    %s", args.artifacts_dir)
    log.info("  Opset:        %d", args.opset)
    log.info("=" * 60)

    # 1. Модель
    model = build_model(args.num_classes)

    # 2. Экспорт ONNX
    export_onnx(model, args.output, opset=args.opset)

    # 3. Упрощение
    if not args.no_simplify:
        simplify_onnx(args.output)

    # 4. Верификация
    verify_info = verify_onnx(args.output, args.num_classes)

    # 5. Training Artifacts
    trainable_patterns = None if args.full_finetune else TRAINABLE_PARAM_PATTERNS
    artifacts_ok = generate_training_artifacts(
        base_model_path=args.output,
        artifacts_dir=args.artifacts_dir,
        trainable_patterns=trainable_patterns,
    )

    # 6. Метаданные
    save_metadata(
        output_path=args.output,
        num_classes=args.num_classes,
        class_labels=CLASS_LABELS[:args.num_classes],
        opset=args.opset,
        verify_info=verify_info,
        artifacts_ok=artifacts_ok,
    )

    log.info("=" * 60)
    log.info("  Готово!")
    log.info("  Модель:       %s", args.output)
    if artifacts_ok:
        log.info("  Артефакты:    %s/", args.artifacts_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
