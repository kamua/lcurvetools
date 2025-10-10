import warnings


def get_mode_by_metric_name(name: str) -> str:
    """
    Get the optimization mode (min or max) for a metric based on its name.
    If the name contains "loss", "error", "hinge", "crossentropy", "false",
    "divergence" or "poisson", it is assumed to be minimized; otherwise,
    it is assumed to be maximized.

    Parameters
    ----------
    name : str
        The name of the metric.

    Returns
    -------
    str
        The mode for the metric ('min' or 'max').

    Examples
    --------
    >>> get_mode_by_metric_name("val_loss")
    'min'

    >>> get_mode_by_metric_name("val_accuracy")
    'max'

    >>> get_mode_by_metric_name("mean_absolute_percentage_error")
    'min'
    """
    if not isinstance(name, str) or len(name) == 0:
        raise TypeError("name must be a non-empty string")

    name = name.lower()
    if name.startswith("val_"):
        name = name[4:]
    if (
        "loss" in name
        or "error" in name
        or "hinge" in name
        or "crossentropy" in name
        or "false" in name
        or "divergence" in name
        or "poisson" in name
    ):
        return "min"
    return "max"


def get_best_epoch_value(
    metric_values: list[float],
    metric_name: str = None,
    mode: str = "auto",
    verbose: bool = True,
) -> tuple[int, float]:
    """
    Get the epoch index and value of the best metric from a list of metric values.

    The best metric is determined based on the specified optimization mode:
    - "auto": Automatically determine if the metric should be minimized or maximized
      based on its name with the `lcurves.utils.get_mode_by_metric_name` function.
    - "min": The metric is minimized (lower values are better).
    - "max": The metric is maximized (higher values are better).

    Parameters
    ----------
    metric_values : list of float
        The values of the metric at each epoch.
    metric_name : str or `None`, default=None
        The name of the metric (used for automatic mode detection).
    mode : str, default="auto"
        The optimization mode for selecting the best epoch ('auto', 'min', or 'max').
    verbose : bool, default=True
        If True, warnings will be issued if the metric does not appear
        to be optimizing as expected.

    Returns
    -------
    tuple of (int, float)
        A tuple containing the best epoch index and the best value.

    Examples
    --------
    >>> get_best_epoch_value([0.5, 0.4, 0.3, 0.35], 'val_loss')
    (2, 0.3)

    >>> get_best_epoch_value([0.55, 0.62, 0.6, 0.57], 'accuracy')
    (1, 0.62)

    >>> get_best_epoch_value([0.5, 0.4, 0.3, 0.35], mode='min')
    (2, 0.3)
    """
    if not isinstance(metric_values, list):
        raise TypeError("metric_values must be a list of floats")
    if len(metric_values) == 0:
        raise ValueError(
            f"metric_values must be a non-empty list but got {metric_values}"
        )
    if len(metric_values) == 1:
        return 0, metric_values[0]
    if metric_name is not None:
        if not isinstance(metric_name, str) or len(metric_name) == 0:
            raise TypeError(
                f"metric_name must be a non-empty string or None but got {metric_name}"
            )
    if mode not in ["auto", "min", "max"]:
        raise ValueError(f"mode must be 'auto', 'min', or 'max' but got {mode}")
    if mode == "auto" and metric_name is None:
        raise ValueError("metric_name must be provided when mode is 'auto'")

    _mode = get_mode_by_metric_name(metric_name) if mode == "auto" else mode

    if _mode == "max":
        best_value = max(metric_values)
        best_epoch = metric_values.index(best_value)
        if verbose and best_value <= metric_values[0]:
            if mode == "auto":
                warnings.warn(
                    (
                        f"The metric '{metric_name}' is detected by its name as being maximized, "
                        "but it appears to not be maximizing.\nConsider using mode='max' or mode='min'."
                    ),
                    UserWarning,
                )
            else:
                warnings.warn(
                    f"\nThe metric seems to not be maximized, but mode='max' was specified.\n"
                    "Check if this is correct.",
                    UserWarning,
                )
        return best_epoch, best_value

    best_value = min(metric_values)
    best_epoch = metric_values.index(best_value)
    if verbose and best_value >= metric_values[0]:
        if mode == "auto":
            warnings.warn(
                f"The metric '{metric_name}' is detected by its name as being minimized, "
                "but it appears to not be minimizing.\nConsider using mode='max' or mode='min'.",
                UserWarning,
            )
        else:
            warnings.warn(
                f"\nThe metric seems to not be minimized, but mode='min' was specified.\n"
                "Check if this is correct.",
                UserWarning,
            )

    return best_epoch, best_value
