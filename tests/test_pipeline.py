"""Testing Pipelines for Logistic Regression and Random Forest"""

# Importing modules.
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from ml_iris_classification.iris_model import get_metrics


# Sample data preparation.
def test_scale_data_mean_std():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Checking if mean ≈ 0 and std. ≈ 1
    assert np.allclose(x_scaled.mean(axis = 0), [0, 0], atol = 1e-9)
    assert np.allclose(x_scaled.std(axis = 0), [1, 1], atol = 1e-9)
    assert x.shape == x_scaled.shape


# Testing built-in function get_metrics.
def test_get_metrics_output_shape():
    y_true = ["setosa", "versicolor", "virginica", "setosa", "virginica"]
    y_pred = ["setosa", "versicolor", "versicolor", "setosa", "virginica"]

    class_names = ["setosa", "versicolor", "virginica"]
    report = classification_report(y_true, y_pred, target_names = class_names, output_dict = True)

    precision, recall, f1 = get_metrics(report)

    assert isinstance(precision, list) and len(precision) == len(class_names)
    assert isinstance(recall, list) and len(recall) == len(class_names)
    assert isinstance(f1, list) and len(f1) == len(class_names)

    for metrics in precision + recall + f1:
        assert 0 <= metrics <= 1


def test_get_metrics_all_classes():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9, "f1-score": 0.95},
        "versicolor": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
        "virginica": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92}
    }

    expected_precision = [1.0, 0.8, 0.9]
    expected_recall = [0.9, 0.7, 0.95]
    expected_f1 = [0.95, 0.75, 0.92]

    precision, recall, f1 = get_metrics(report)

    assert precision == expected_precision
    assert recall == expected_recall
    assert f1 == expected_f1


def test_get_metrics_virgi_missing():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9, "f1-score": 0.95},
        "versicolor": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "virginica" in str(excinfo.value)


def test_get_metrics_setosa_missing():
    report = {
        "versicolor": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
        "virginica": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92}
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "setosa" in str(excinfo.value)


def test_get_metrics_versi_missing():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9, "f1-score": 0.95},
        "virginica": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92}
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "versicolor" in str(excinfo.value)