"""Tests for plotting functionality in lcurvetools."""

import numpy as np
import matplotlib.pyplot as plt
from lcurvetools import (
    lcurves,
    lcurves_by_MLP_estimator,
    history_concatenate,
)


def test_lcurves_by_history_basic(simple_history):
    """Test basic learning curve plotting with a single history."""
    axes = lcurves(simple_history)
    assert len(axes) == 3  # Loss, metrics and lr axes
    plt.close()


def test_lcurves_by_history_metrics(simple_history):
    """Test learning curve plotting with specific metrics."""
    metrics = ["accuracy"]
    axes = lcurves(simple_history, plot_metrics=metrics)
    assert len(axes) == 3  # Loss, single metric and lr axes
    plt.close()


def test_lcurves_by_history_no_validation(simple_history):
    """Test plotting without validation metrics and learning rate."""
    history = {
        "loss": simple_history["loss"],
        "accuracy": simple_history["accuracy"],
    }
    axes = lcurves(history)
    assert len(axes) == 2
    plt.close()


def test_lcurves_by_history_with_lr(simple_history):
    """Test learning curve plotting with learning rate."""
    axes = lcurves(simple_history, plot_learning_rate=True)
    assert len(axes) == 3  # Loss, metrics, and lr axes
    plt.close()


def test_lcurves_by_mlp_estimator_basic(mock_mlp_estimator):
    """Test basic MLP estimator learning curve plotting."""
    ax = lcurves_by_MLP_estimator(mock_mlp_estimator)
    assert isinstance(ax[0], plt.Axes)
    plt.close()


def test_history_concatenate(multi_model_histories):
    """Test history concatenation functionality."""
    concatenated = history_concatenate(
        multi_model_histories[0], multi_model_histories[1]
    )

    # Check structure
    assert all(
        key in concatenated
        for key in ["loss", "val_loss", "accuracy", "val_accuracy"]
    )

    # Check lengths
    expected_length = sum(len(h["loss"]) for h in multi_model_histories)
    assert len(concatenated["loss"]) == expected_length

    # Check values
    expected_loss = (
        multi_model_histories[0]["loss"] + multi_model_histories[1]["loss"]
    )
    np.testing.assert_array_equal(concatenated["loss"], expected_loss)


def test_history_concatenate_missing_metrics(multi_model_histories):
    """Test history concatenation with missing metrics."""
    histories = [
        {"loss": [0.5, 0.3], "accuracy": [0.7, 0.8]},
        {"loss": [0.4, 0.2], "val_loss": [0.45, 0.25]},
    ]
    concatenated = history_concatenate(histories[0], histories[1])
    assert len(concatenated["loss"]) == 4
    assert len(concatenated["val_loss"]) == 4
    assert len(concatenated["accuracy"]) == 4
