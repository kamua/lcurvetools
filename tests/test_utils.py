import pytest

from lcurvetools.utils import get_mode_by_metric_name, get_best_epoch_value


def test_get_mode_by_metric_name_basic():
    assert get_mode_by_metric_name("val_loss") == "min"
    assert get_mode_by_metric_name("accuracy") == "max"
    assert get_mode_by_metric_name("poisson_divergence") == "min"


def test_get_best_epoch_value_min_mode():
    vals = [0.5, 0.4, 0.3, 0.35]
    idx, val = get_best_epoch_value(vals, mode="min")
    assert idx == 3
    assert val == pytest.approx(0.3)


def test_get_best_epoch_value_max_mode():
    vals = [0.55, 0.62, 0.6, 0.57]
    idx, val = get_best_epoch_value(vals, mode="max")
    assert idx == 1
    assert val == pytest.approx(0.62)


def test_get_best_epoch_value_auto_mode_by_name():
    vals = [0.5, 0.4, 0.3, 0.35]
    idx, val = get_best_epoch_value(vals, metric_name="val_loss", mode="auto")
    assert idx == 2
    assert val == pytest.approx(0.3)


def test_get_best_epoch_value_single_element():
    vals = [0.123]
    idx, val = get_best_epoch_value(vals, mode="min")
    assert idx == 0
    assert val == pytest.approx(0.123)


def test_get_best_epoch_value_empty_raises():
    with pytest.raises(ValueError):
        get_best_epoch_value([], mode="min")


def test_get_best_epoch_value_auto_requires_metric_name():
    with pytest.raises(ValueError):
        get_best_epoch_value([0.1, 0.2], mode="auto")


def test_get_best_epoch_value_warns_when_not_optimizing_max():
    # For a metric detected as maximized but best value equals the first value
    vals = [0.6, 0.59, 0.58]
    with pytest.warns(UserWarning):
        get_best_epoch_value(vals, metric_name="accuracy", mode="auto")


def test_get_best_epoch_value_warns_when_not_optimizing_min():
    # For a metric detected as minimized but best value equals the first value
    vals = [0.6, 0.61, 0.62]
    with pytest.warns(UserWarning):
        get_best_epoch_value(vals, metric_name="val_loss", mode="auto")
