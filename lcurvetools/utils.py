import warnings


def get_mode_by_metric_name(name: str):
    name = name.lower()
    if name.startswith("val_"):
        name = name[4:]
    if (
        name.startswith("loss")
        or "_loss" in name
        or "loss_" in name
        or "error" in name
    ):
        return "min"
    return "max"


def get_best_epoch_value(
    metric_name: str, metric_values: list[float], mode: str = "auto"
):
    if not isinstance(metric_name, str):
        raise TypeError("metric_name must be a string")
    if not isinstance(metric_values, list):
        raise TypeError("metric_values must be a list of floats")
    if len(metric_values) == 0:
        raise ValueError("metric_values must be a non-empty list")
    if len(metric_values) == 1:
        return 0, metric_values[0]
    if mode not in ["auto", "min", "max"]:
        raise ValueError("mode must be 'auto', 'min', or 'max'")

    _mode = get_mode_by_metric_name(metric_name) if mode == "auto" else mode

    if _mode == "max":
        best_value = max(metric_values)
        best_epoch = metric_values.index(best_value)
        if best_epoch <= metric_values.index(min(metric_values)):
            if mode == "auto":
                raise UserWarning(
                    f"The metric '{metric_name}' is detected by its name as being maximized, but it appears to not be maximizing.\nConsider using mode='max' or mode='min'."
                )
            else:
                warnings.warn(
                    f"\nThe metric '{metric_name}' seems to not be maximized, but mode='max' was specified.\nCheck if this is correct.",
                    UserWarning,
                )
        return best_epoch, best_value

    best_value = min(metric_values)
    best_epoch = metric_values.index(best_value)
    if best_epoch <= metric_values.index(max(metric_values)):
        if mode == "auto":
            raise UserWarning(
                f"The metric '{metric_name}' is detected by its name as being minimized, but it appears to not be minimizing.\nConsider using mode='max' or mode='min'."
            )
        else:
            warnings.warn(
                f"\nThe metric '{metric_name}' seems to not be minimized, but mode='min' was specified.\nCheck if this is correct.",
                UserWarning,
            )

    return best_epoch, best_value
