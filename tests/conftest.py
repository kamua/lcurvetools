"""Test configuration for lcurvetools."""

from sklearn.neural_network import MLPClassifier
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import pytest
import numpy as np


@pytest.fixture
def simple_history():
    """Single model training history with loss and accuracy."""
    return {
        "loss": [0.5, 0.3, 0.2, 0.15],
        "val_loss": [0.6, 0.4, 0.25, 0.2],
        "accuracy": [0.7, 0.8, 0.85, 0.9],
        "val_accuracy": [0.65, 0.75, 0.8, 0.85],
        "lr": [0.01, 0.01, 0.001, 0.001],
    }


@pytest.fixture
def multi_model_histories():
    """Multiple model training histories for comparison."""
    return [
        {
            "loss": [0.5, 0.3, 0.2],
            "val_loss": [0.6, 0.4, 0.25],
            "accuracy": [0.7, 0.8, 0.85],
            "val_accuracy": [0.65, 0.75, 0.8],
        },
        {
            "loss": [0.45, 0.25, 0.15],
            "val_loss": [0.55, 0.35, 0.2],
            "accuracy": [0.75, 0.85, 0.9],
            "val_accuracy": [0.7, 0.8, 0.85],
        },
    ]


@pytest.fixture
def mock_mlp_estimator():
    """Mock MLPClassifier with loss curve and validation scores."""

    class MockMLPEstimator(MLPClassifier):
        def __init__(self):
            self.loss_curve_ = [0.5, 0.3, 0.2, 0.15]
            self.validation_scores_ = [0.7, 0.8, 0.85, 0.9]

    return MockMLPEstimator()
