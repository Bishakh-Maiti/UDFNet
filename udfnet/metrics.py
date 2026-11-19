"""OA, AA, Kappa and per-class accuracy, as reported in the paper's Tables 5-8."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = confusion(np.asarray(y_true), np.asarray(y_pred), n_classes)
    total = cm.sum()
    per_class = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    present = cm.sum(1) > 0
    oa = np.trace(cm) / max(total, 1)
    pe = (cm.sum(0) * cm.sum(1)).sum() / max(total, 1) ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "OA": 100 * oa,
        "AA": 100 * per_class[present].mean(),
        "Kappa": 100 * kappa,  # reported x100, like the paper
        "per_class": (100 * per_class).tolist(),
    }
