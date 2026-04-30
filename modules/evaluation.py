"""
Evaluation Module
Calculates detection performance metrics and generates comparison reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_supervised(y_true, y_pred, labels=None):
    """
    Evaluate supervised detection (Random Forest) performance.
    """
    # Handle pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    report = classification_report(y_true, y_pred, zero_division=0)

    return metrics, cm, report


def evaluate_unsupervised(y_true, y_pred_anomaly):
    """
    Evaluate unsupervised detection (Isolation Forest) performance.
    """
    # Handle pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values

    # Map true labels: BENIGN -> Normal, everything else -> Suspicious
    y_true_mapped = np.array([
        "Normal" if str(label).upper() == "BENIGN" else "Suspicious"
        for label in y_true
    ])

    metrics = {
        "accuracy": accuracy_score(y_true_mapped, y_pred_anomaly),
        "precision": precision_score(
            y_true_mapped, y_pred_anomaly, pos_label="Suspicious",
            average="binary", zero_division=0
        ),
        "recall": recall_score(
            y_true_mapped, y_pred_anomaly, pos_label="Suspicious",
            average="binary", zero_division=0
        ),
        "f1_score": f1_score(
            y_true_mapped, y_pred_anomaly, pos_label="Suspicious",
            average="binary", zero_division=0
        ),
    }

    cm = confusion_matrix(y_true_mapped, y_pred_anomaly, labels=["Normal", "Suspicious"])

    return metrics, cm


def compare_methods(rf_metrics, if_metrics):
    """
    Create a side-by-side comparison of Random Forest and Isolation Forest metrics.
    """
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest": [
            rf_metrics.get("accuracy", 0),
            rf_metrics.get("precision", 0),
            rf_metrics.get("recall", 0),
            rf_metrics.get("f1_score", 0),
        ],
        "Isolation Forest": [
            if_metrics.get("accuracy", 0),
            if_metrics.get("precision", 0),
            if_metrics.get("recall", 0),
            if_metrics.get("f1_score", 0),
        ],
    })

    return comparison
