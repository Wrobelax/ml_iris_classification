"""Testing Pipelines for Logistic Regression and Random Forest"""

# Importing modules.
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from ml_iris_classification.iris_model import get_metrics



"""Testing basic data"""
# Sample data preparation.
def test_scale_data_mean_std():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Checking if mean ≈ 0 and std. ≈ 1
    assert np.allclose(x_scaled.mean(axis = 0), [0, 0], atol = 1e-9)
    assert np.allclose(x_scaled.std(axis = 0), [1, 1], atol = 1e-9)
    assert x.shape == x_scaled.shape


# Checking if LabelEncoder works properly.
def test_label_encoder():
    labels = ["setosa", "versicolor", "virginica"]
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    decoded = le.inverse_transform(encoded)

    assert list(decoded) == labels


"""Logistic Regression"""
# Testing if Logistic Regression learns without error.
def     test_logreg_pipeline_fit_without_error():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter = 200, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5]]
    y = [0, 1, 0]

    pipeline.fit(x, y)


# Testing if pipeline predictions are correct.
def test_logreg_pipeline_pred_shape():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter = 200, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5]]
    y = [0, 1, 0]

    pipeline.fit(x, y)
    preds = pipeline.predict(x)

    assert len(preds) == len(x)


# Testing pipeline for cross-validation.
def test_logreg_pipeline_cross_val():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter = 200, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5], [0.6, 0.8]]
    y = [0, 1, 0, 1]

    scores = cross_val_score(pipeline, x, y, cv = 2)

    assert len(scores) == 2
    assert all(0 <= score <= 1 for score in scores)


# Testing iris data.
def test_logreg_pipeline_on_iris_data():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter = 200))
    ])
    iris = load_iris()
    x, y = iris.data, iris.target

    pipeline.fit(x,y)
    preds = pipeline.predict(x)

    assert len(preds) == len(x)
    assert set(preds).issubset(set(y))


# Testing invalid data input.
def test_logreg_pipeline_invalid_data():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    x_invalid = [["a", "b"],["c", "d"]]
    y = [0, 1]

    with pytest.raises(ValueError):
        pipeline.fit(x_invalid, y)



"""Random Forest"""
# Testing if Random Forest learns without error.
def test_randfor_pipeline_fit_without_error():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5]]
    y = [0, 1, 0]

    pipeline.fit(x, y)


# Testing if pipeline predictions are correct.
def test_randfor_pipeline_pred_shape():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5]]
    y = [0, 1, 0]

    pipeline.fit(x, y)
    preds = pipeline.predict(x)

    assert len(preds) == len(x)


# Testing pipeline for cross-validation.
def test_randfor_pipeline_cross_val():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100, random_state = 42))
    ])
    x = [[0.1, 0.2], [0.3, 0.1], [0.4, 0.5], [0.6, 0.8]]
    y = [0, 1, 0, 1]

    scores = cross_val_score(pipeline, x, y, cv = 2)

    assert len(scores) == 2
    assert all(0 <= score <= 1 for score in scores)


# Testing iris data.
def test_randfor_pipeline_on_iris_data():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100, random_state = 42))
    ])
    iris = load_iris()
    x, y = iris.data, iris.target

    pipeline.fit(x,y)
    preds = pipeline.predict(x)

    assert len(preds) == len(x)
    assert set(preds).issubset(set(y))


# Testing invalid data input.
def test_randfor_pipeline_invalid_data():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier())
    ])
    x_invalid = [["a", "b"],["c", "d"]]
    y = [0, 1]

    with pytest.raises(ValueError):
        pipeline.fit(x_invalid, y)



"""Get_metrics function"""
# Testing if all get_metrics outputs are present.
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


# Testing if all classes are present.
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


# Testing if KeyError is returned when some data is missing.
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


def test_get_metrics_partially_missing_f1():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9},
        "versicolor": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
        "virginica": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92}
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "f1-score" in str(excinfo.value)


def test_get_metrics_partially_missing_precision():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9, "f1-score": 0.95},
        "versicolor": {"recall": 0.7, "f1-score": 0.75},
        "virginica": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92}
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "precision" in str(excinfo.value)


def test_get_metrics_partially_missing_recall():
    report = {
        "setosa": {"precision": 1.0, "recall": 0.9, "f1-score": 0.95},
        "versicolor": {"precision": 0.7, "recall": 0.7, "f1-score": 0.75},
        "virginica": {"precision": 0.9, "f1-score": 0.92}
    }

    with pytest.raises(KeyError) as excinfo:
        get_metrics(report)

    assert "recall" in str(excinfo.value)


# Testing if KeyError is returned for an empty report.
def test_get_metrics_empty_report():
    report = {}

    with pytest.raises(KeyError):
        get_metrics(report)