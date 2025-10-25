"""Tests for plotting functionality in lcurvetools."""

import numpy as np
import matplotlib.pyplot as plt
from lcurvetools import lcurvetools as lct


def test_lcurves_by_history_basic(simple_history):
    """Test basic learning curve plotting with a single history."""
    fig, axes = lct.lcurves_by_history(simple_history)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2  # Loss and metrics axes
    plt.close(fig)


def test_lcurves_by_history_metrics(simple_history):
    """Test learning curve plotting with specific metrics."""
    metrics = ["accuracy"]
    fig, axes = lct.lcurves_by_history(simple_history, metrics=metrics)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2  # Loss and single metric
    plt.close(fig)


def test_lcurves_by_history_no_validation(simple_history):
    """Test plotting without validation metrics."""
    history = {
        "loss": simple_history["loss"],
        "accuracy": simple_history["accuracy"],
    }
    fig, axes = lct.lcurves_by_history(history)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)


def test_lcurves_by_history_with_lr(simple_history):
    """Test learning curve plotting with learning rate."""
    fig, axes = lct.lcurves_by_history(simple_history, show_lr=True)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 3  # Loss, metrics, and lr axes
    plt.close(fig)


def test_lcurves_by_mlp_estimator_basic(mock_mlp_estimator):
    """Test basic MLP estimator learning curve plotting."""
    fig, ax = lct.lcurves_by_MLP_estimator(mock_mlp_estimator)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_lcurves_by_mlp_estimator_custom_title(mock_mlp_estimator):
    """Test MLP estimator plotting with custom title."""
    title = "Custom MLP Learning Curves"
    fig, ax = lct.lcurves_by_MLP_estimator(mock_mlp_estimator, title=title)
    assert ax.get_title() == title
    plt.close(fig)


def test_history_concatenate(multi_model_histories):
    """Test history concatenation functionality."""
    concatenated = lct.history_concatenate(multi_model_histories)

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
    concatenated = lct.history_concatenate(histories)
    assert len(concatenated["loss"]) == 4
    assert "val_loss" in concatenated
    assert "accuracy" in concatenated
