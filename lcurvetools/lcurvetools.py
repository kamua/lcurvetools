import warnings
from copy import deepcopy
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

from .utils import get_best_epoch_value


def _get_n_epochs(history: Mapping) -> int:
    """Return number of epochs (length of value lists) for a history dict.

    Raises TypeError if not all values have the same length.
    """
    n_epochs = set(map(len, history.values()))
    if len(n_epochs) != 1:
        raise TypeError(
            "The values of all `history` keys should be lists of the same "
            "length, equal to the number of epochs."
        )
    return list(n_epochs)[0]


def _get_train_key_names(keys: Sequence[str]) -> list[str]:
    """Normalize key names by stripping leading 'val_' and return unique names."""
    train_keys: list[str] = []
    for key in keys:
        train_keys.append(key[4:] if key.startswith("val_") else key)
    return list(set(train_keys))


def _get_colors(num: int) -> list[tuple[float, float, float]]:
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    cmap = plt.get_cmap("tab10").colors
    if num <= 10:
        return cmap[:num]
    cmap += plt.get_cmap("tab20b").colors[::4]
    if num <= 15:
        return cmap[:num]
    cmap += plt.get_cmap("tab20c").colors[1::4]
    if num <= 20:
        return cmap[:num]
    cmap += plt.get_cmap("tab20b").colors[2::4]
    if num <= 25:
        return cmap[:num]
    cmap += plt.get_cmap("tab20").colors[1::2]
    if num <= 35:
        return cmap[:num]
    cmap += plt.get_cmap("tab20b").colors[1::4]
    if num <= 40:
        return cmap[:num]
    cmap += plt.get_cmap("tab20c").colors[2::4]
    if num <= 45:
        return cmap[:num]
    cmap += plt.get_cmap("tab20b").colors[3::4]
    if num <= 50:
        return cmap[:num]
    cmap += plt.get_cmap("tab20c").colors[::4]
    if num <= 55:
        return cmap[:num]
    cmap += plt.get_cmap("tab20c").colors[3::4]
    if num <= 60:
        return cmap[:num]

    while len(cmap) < num:
        cmap += cmap
    return cmap[:num]


def lcurves_by_history(
    histories: dict | list[dict],
    initial_epoch: int = 0,
    epoch_range_to_scale: int | list[int] | tuple[int, int] = 0,
    plot_losses: bool | list[str] = True,
    plot_metrics: bool | list[str] = True,
    plot_learning_rate: bool | list[str] = True,
    color_grouping_by: str | None = None,
    model_names: list[str] = None,
    optimization_modes: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """
    Plots learning curves of a neural network model trained with the keras
    framework. Dependences of values of the losses, metrics and the learning
    rate on the epoch index can be plotted on three subplots along a figure
    column. The best values are marked for dependencies of losses and metrics
    (minimum values for losses and best values for metrics).

    Parameters
    ----------
    histories : dict or list of dict
        - If dict, it could contain keys with training and validation values
        of losses and metrics, as well as learning rate values at successive
        epochs in the format of the `history` attribute of the `History`
        object which is returned by the
        [fit](https://keras.io/api/models/model_training_apis/#fit-method)
        method of the model. The values of all keys should be represented by
        numeric lists of the same length, equaled to the number of epochs
        `n_epochs`.
        - If list of dict, each dict in the list should be in the same format
        as described above for a single dict. The list of dicts is treated as
        a collection of fitting histories, and the plots will display all
        training and validation curves for each history.

    initial_epoch : int, default=0
        The epoch index at which the `fit` method had started to train
        the model. The parameter corresponds to the same parameter of the
        [fit](https://keras.io/api/models/model_training_apis/#fit-method)
        method of a keras model. Also, setting `initial_epoch=1` can be useful
        to convert the epoch index plotted along the horizontal axes of the
        subplots into the number of passed epochs.

    epoch_range_to_scale : int or list (tuple) of int or None, default=0
        Specifies the epoch index range within which the subplots of the
        losses and metrics are scaled.
        - If `epoch_range_to_scale` is a list or a tuple of two int values,
        then they specify the epoch index limits of the scaling range in the
        form `[start, stop)`, i.e. as for `slice` and `range` objects. If
        `start` is `None`, the scaling range starts from the first epoch. If
        `stop` is `None`, the scaling range ends at the last epoch.
        - If `epoch_range_to_scale` is an int value, then it specifies the
        lower epoch index `start` of the scaling range, and the losses and
        metrics subplots are scaled by epochs with indices from `start` to the
        last. This case is equivalent to `epoch_range_to_scale = [start, None]`.

        The epoch index values `start`, `stop` must take into account
        the value of the `initial_epoch` parameter.

    plot_losses : bool or list of str, default=True
        - If bool, it specifies the need to plot a subplot of losses.
        Dictionary keys with the name "loss" and names containing the
        substring "_loss" are treated as losses keys.
        - If list, it specifies loss key names of the `histories` dictionaries
        that should be plotted into the losses subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        'val_' of the specified key names.

    plot_metrics : bool or list of str, default=True
        - If bool, it specifies the need to plot a subplot of metrics.
        Dictionary keys that have not been recognized as loss or learning rate
        keys are treated as metrics keys.
        - If list, it specifies metric key names of the `histories` dictionaries
        that should be plotted into the metrics subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        'val_' of the specified key names.

    plot_learning_rate : bool or list of str, default=True
        - If bool, it specifies the need to plot a subplot of learning rate.
        Dictionary keys with the name "lr" and names containing the
        substring "learning_rate" are treated as learning rate keys.
        - If list, it specifies learning rate key names of the `histories`
        dictionaries that should be plotted into the learning rate subplot.

        Learning rate values on the vertical axis are plotted in a logarithmic
        scale.

    color_grouping_by : str or None, default=None
        Specifies how colors of curves in the subplots are grouped.
        - If 'model', all curves corresponding to a single model history
        (a single dictionary in the `histories` list) are plotted in the same
        color.
        - If 'quantity', all curves corresponding to a single quantity (loss,
        metric, learning rate) inside a subplot are plotted in the same color.
        - If None, each pair of curves for the training and validation subsets
        is plotted in one unique color but with different line styles.

        If the learning history of one model is given for one loss function and
        one metric, then pairs of learning curves for the training and
        validation subsets are plotted with lines of the same style but
        different colors, regardless of the specified value of the
        `color_grouping_by` parameter.

    model_names : list of str or None, default=None
        Specifies model names for each history in the `histories` list. The names
        will be used in the legends of the subplots if `unique_curve_colors` is
        True. The length of the `model_names` list must be equal to the length
        of the `histories` list.
        If `None`, the function will use default names.

    optimization_modes : dict or None, default=None
        - If dict, it specifies optimization modes for each metric name in the
        `histories` dictionaries. For example, if
        `optimization_modes = {"iou": "max", "hinge": "min"}`
        the function will consider iou as metric for maximization
        and hinge as metric for minimization. If a metric name is not found in
        the `optimization_modes` dictionary, the optimization mode will be
        determined automatically based on the metric name with the function
        `lcurvetools.utils.get_mode_by_metric_name`.
        - If `None`, the "auto" mode will be used for all metrics.
        It only affects the marking of the best values on the subplot of metrics.

    figsize : a tuple (width, height) in inches or `None`, default=None.
        Specifies size of creating figure. If `None`, default values of width
        and height of a figure for the matplotlib library will be used.

    Returns
    -------
    numpy array or list of `matplotlib.axes.Axes` object
        Each `matplotlib.axes.Axes` object in the numpy array or list
        corresponds to the built subplot from top to bottom.

    Examples
    --------
    >>> import keras
    >>> from lcurvetools import lcurves_by_history

    [Create](https://keras.io/api/models/), [compile](https://keras.io/api/models/model_training_apis/#compile-method)
    and [fit](https://keras.io/api/models/model_training_apis/#fit-method) the keras model:

    >>> model = keras.Model(...) # or keras.Sequential(...)
    >>> model.compile(...)
    >>> hist = model.fit(...)

    Use `hist.history` dictionary to plot the learning curves as the
    dependences of values of all keys in the dictionary on an epoch
    index with automatic recognition of keys of losses, metrics and
    learning rate:

    >>> lcurves_by_history(hist.history);

    If the model is fitted multiple times, then all the fitting histories
    can be plotted in a single figure.

    >>> histories = []
    >>> for i in range(5):
    >>>     model = keras.Model(...) # or keras.Sequential(...)
    >>>     model.compile(...)
    >>>     hist = model.fit(...)
    >>>     histories.append(hist.history)
    >>> lcurves_by_history(histories);
    """

    def get_ylims(keys):
        ylim_top = -float("inf")
        ylim_bottom = float("inf")
        for hist in histories:
            for key in keys:
                if key not in hist.keys():
                    continue
                ylim_top = max(ylim_top, max(hist[key][epochs_slice]))
                ylim_bottom = min(ylim_bottom, min(hist[key][epochs_slice]))
        pad = (ylim_top - ylim_bottom) * 0.05
        if pad == 0:
            pad = 0.01
        return dict(bottom=ylim_bottom - pad, top=ylim_top + pad)

    def get_plot_keys(plot_, _keys):
        if isinstance(plot_, (list, tuple)):
            if plot_:
                keys: list[str] = []
                for hist in histories:
                    for key_name in plot_:
                        if key_name in hist:
                            keys.append(key_name)
                    keys += [
                        "val_" + key_name
                        for key_name in plot_
                        if "val_" + key_name in hist
                    ]
                return sorted(set(keys))
        if plot_:
            return sorted(_keys)
        return []

    # Input data validation

    if not isinstance(histories, (list, dict)):
        raise TypeError(
            "The `histories` parameter should be a dict or a list of dicts."
        )
    if not histories:
        raise ValueError("The `histories` list or dictionary cannot be empty.")
    if isinstance(histories, list):
        for i, hist in enumerate(histories):
            if not isinstance(hist, dict):
                raise TypeError(
                    f"The {i}-th element of the `histories` list is not a dict."
                )
            if not hist:
                raise ValueError(
                    f"The {i}-th dictionary in the `histories` list cannot be"
                    " empty."
                )
    if isinstance(histories, dict):
        histories = [histories]
    n_epochs = [_get_n_epochs(hist) for hist in histories]
    n_epochs_max = max(n_epochs)

    if epoch_range_to_scale is None:
        epochs_slice = slice(0, n_epochs_max)
    elif isinstance(epoch_range_to_scale, int):
        epochs_slice = slice(
            max(0, epoch_range_to_scale - initial_epoch), n_epochs_max
        )
    elif (
        isinstance(epoch_range_to_scale, (list, tuple))
        and len(epoch_range_to_scale) == 2
    ):
        if epoch_range_to_scale[0] is None:
            start_epoch_index = 0
        else:
            start_epoch_index = max(0, epoch_range_to_scale[0] - initial_epoch)
        if epoch_range_to_scale[1] is None:
            stop_epoch_index = n_epochs_max
        else:
            stop_epoch_index = min(
                n_epochs_max,
                max(1, epoch_range_to_scale[1] - initial_epoch + 1),
            )
        epochs_slice = slice(
            start_epoch_index,
            stop_epoch_index,
        )
    else:
        raise TypeError(
            "The `epoch_range_to_scale` parameter should be an int or a"
            " list/tuple of two ints."
        )

    if not isinstance(plot_losses, (bool, list, tuple)):
        raise TypeError(
            "The `plot_losses` parameter should be bool, list or tuple"
        )
    if not isinstance(plot_metrics, (bool, list, tuple)):
        raise TypeError(
            "The `plot_metrics` parameter should be bool, list or tuple"
        )
    if not isinstance(plot_learning_rate, (bool, list, tuple)):
        raise TypeError(
            "The `plot_learning_rate` parameter should be bool, list or tuple"
        )

    if optimization_modes is not None:
        if not isinstance(optimization_modes, dict):
            raise TypeError(
                "The `optimization_modes` parameter should be a dict or None."
            )
        for lr_name, value in optimization_modes.items():
            if not isinstance(lr_name, str) or not lr_name:
                raise TypeError(
                    "The keys of the `optimization_modes` dict should be"
                    " non-empty strings."
                )
            if value not in ("min", "max"):
                raise ValueError(
                    "The values of the `optimization_modes` dict should be"
                    " 'min' or 'max'."
                )
    if not isinstance(initial_epoch, int) or initial_epoch < 0:
        raise ValueError(
            "The `initial_epoch` parameter should be a non-negative integer."
        )
    if model_names is not None:
        if not isinstance(model_names, list):
            raise TypeError(
                "The `model_names` parameter should be a list of strings or"
                " None."
            )
        if len(model_names) != len(histories):
            raise ValueError(
                "The length of `model_names` must equal the length of"
                " `histories`."
            )
        for name in model_names:
            if not isinstance(name, str):
                raise TypeError(
                    "Each element of `model_names` must be a string."
                )
    elif len(histories) > 1:
        model_names = [str(i) for i in range(len(histories))]
    else:
        model_names = [""]

    if color_grouping_by not in ("model", "metric", None):
        raise ValueError(
            "The `color_grouping_by` parameter should be 'model', 'metric', or"
            " None."
        )
    # End of input data validation

    # Extract keys for losses, learning rates, and metrics
    loss_keys = []
    lr_keys = []
    metric_keys = []
    for hist in histories:
        loss_keys += [
            name for name in hist.keys() if name == "loss" or "_loss" in name
        ]
        lr_keys += [
            name
            for name in hist.keys()
            if "lr" == name or "learning_rate" in name
        ]
        metric_keys += [
            name for name in hist.keys() if name not in (loss_keys + lr_keys)
        ]
    loss_keys = list(set(loss_keys))
    lr_keys = list(set(lr_keys))
    metric_keys = list(set(metric_keys))

    plot_loss_keys = get_plot_keys(plot_losses, loss_keys)
    n_subplots = int(len(plot_loss_keys) > 0)

    metric_keys = [key for key in metric_keys if key not in plot_loss_keys]
    plot_metric_keys = get_plot_keys(plot_metrics, metric_keys)
    n_subplots += int(len(plot_metric_keys) > 0)

    lr_keys = [
        key for key in lr_keys if key not in plot_loss_keys + plot_metric_keys
    ]
    plot_lr_keys = get_plot_keys(plot_learning_rate, lr_keys)
    n_subplots += int(len(plot_lr_keys) > 0)

    val_key_exists = False
    for lr_name in plot_loss_keys + plot_metric_keys:
        if lr_name[:4] == "val_":
            val_key_exists = True
            break

    # determine colors of curves in the subplots of losses, metrics and learning rates
    train_loss_names = _get_train_key_names(plot_loss_keys)
    train_metric_names = _get_train_key_names(plot_metric_keys)
    lr_names = _get_train_key_names(plot_lr_keys)

    if len(model_names) < 2 and color_grouping_by == "model":
        color_grouping_by = None
    if (
        len(train_loss_names) < 2
        and len(train_metric_names) < 2
        and len(lr_names) < 2
        and color_grouping_by == "metric"
    ):
        color_grouping_by = None

    simple_color_grouping = (
        len(model_names) < 2
        and len(train_loss_names) < 2
        and len(train_metric_names) < 2
    )

    prefixes = [""]
    linestyles = {prefixes[0]: "-"}
    if val_key_exists:
        prefixes.append("val_")
        linestyles[prefixes[-1]] = (
            linestyles[prefixes[0]] if simple_color_grouping else (0, (2, 1))
        )
    if not simple_color_grouping:
        index = pd.MultiIndex.from_product(
            [train_loss_names, model_names],
            names=["loss", "model"],
        )
        loss_clrs = pd.Series(index=index, dtype="object")
        index = pd.MultiIndex.from_product(
            [train_metric_names, model_names],
            names=["metric", "model"],
        )
        metric_clrs = pd.Series(index=index, dtype="object")
        index = pd.MultiIndex.from_product(
            [lr_names, model_names], names=["lr", "model"]
        )
        lr_clrs = pd.Series(index=index, dtype="object")

        if color_grouping_by is None:
            cmap = _get_colors(
                max(len(loss_clrs), len(metric_clrs), len(lr_clrs))
            )
            loss_clrs.iloc[:] = cmap[: len(loss_clrs)]
            metric_clrs.iloc[:] = cmap[: len(metric_clrs)]
            lr_clrs.iloc[:] = cmap[: len(lr_clrs)]
        elif color_grouping_by == "metric":
            cmap = _get_colors(len(train_loss_names))
            for model in model_names:
                loss_clrs.loc[:, model] = cmap
            cmap = _get_colors(len(train_metric_names))
            for model in model_names:
                metric_clrs.loc[:, model] = cmap
            cmap = _get_colors(len(lr_names))
            for model in model_names:
                lr_clrs.loc[:, model] = cmap
        else:  # color_grouping_by == "model":
            cmap = _get_colors(len(model_names))
            for loss_name in train_loss_names:
                loss_clrs.loc[loss_name] = cmap
            for lr_name in train_metric_names:
                metric_clrs.loc[lr_name] = cmap
            for lr_name in lr_names:
                lr_clrs.loc[lr_name] = cmap

    # Check if we need to scale the y-axis specified by epoch_range_to_scale
    need_to_scale = 0 < epochs_slice.start or epochs_slice.stop < n_epochs_max

    # Create the figure and subplots
    fig = plt.figure(figsize=figsize)
    if n_subplots > 1:
        if n_subplots == 2:
            axs = fig.subplots(n_subplots, 1, sharex=True)
        else:
            axs = fig.subplots(
                n_subplots, 1, sharex=True, height_ratios=[2, 2, 1]
            )
    else:
        axs = [plt.gca()]

    # Configure the subplots
    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(
            axis="x",
            which="both",
            direction="in",
            bottom=True,
            top=True,
        )
        ax.tick_params(
            axis="y",
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            right=True,
        )
        ax.yaxis.set_label_position("left")
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.xaxis.set_minor_locator(ticker.MaxNLocator(steps=[1, 4, 10]))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid()

    axs[-1].tick_params(axis="x", labelbottom=True)
    axs[-1].set_xlabel("epoch")

    x = range(initial_epoch, initial_epoch + n_epochs_max)

    index_subplot = 0
    kwargs_legend = dict(loc="upper left", bbox_to_anchor=(1.002, 1.05))
    markersize = 5
    if len(plot_loss_keys) > 0:
        ax = axs[index_subplot]
        n_labels = 0
        for i, hist in enumerate(histories):
            for lr_name in train_loss_names:
                color = (
                    None
                    if simple_color_grouping
                    else loss_clrs[lr_name, model_names[i]]
                )
                for prefix in prefixes:
                    _key = prefix + lr_name
                    if _key in hist.keys():
                        _label = _key
                        if model_names[i] != "":
                            _label += "_" + model_names[i]
                        if _label is not None:
                            n_labels += 1
                        lines = ax.plot(
                            x[: len(hist[_key])],
                            hist[_key],
                            label=_label,
                            color=color,
                            linestyle=linestyles[prefix],
                        )
                        best_epoch, best_value = get_best_epoch_value(
                            hist[_key], _key, mode="min", verbose=False
                        )
                        ax.plot(
                            x[best_epoch],
                            best_value,
                            marker="o",
                            markersize=markersize,
                            fillstyle="full",
                            color=lines[-1].get_color(),
                        )
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_loss_keys))
        ax.set_ylabel("loss")
        fontsize = min(10, max(5, 16 - n_labels))
        ncol = min(4, 1 + n_labels // 11)
        ax.legend(fontsize=fontsize, ncol=ncol, **kwargs_legend)
        index_subplot += 1

    if len(plot_metric_keys) > 0:
        ax = axs[index_subplot]
        n_labels = 0
        for i, hist in enumerate(histories):
            for lr_name in train_metric_names:
                color = (
                    None
                    if simple_color_grouping
                    else metric_clrs[lr_name, model_names[i]]
                )
                for prefix in prefixes:
                    _key = prefix + lr_name
                    if _key in hist.keys():
                        _label = _key
                        if model_names[i] != "":
                            _label += "_" + model_names[i]
                        if _label is not None:
                            n_labels += 1
                        lines = ax.plot(
                            x[: len(hist[_key])],
                            hist[_key],
                            label=_label,
                            color=color,
                            linestyle=linestyles[prefix],
                        )
                        mode = (
                            optimization_modes[lr_name]
                            if optimization_modes
                            else "auto"
                        )
                        best_epoch, best_value = get_best_epoch_value(
                            hist[_key], _key, mode=mode, verbose=False
                        )
                        ax.plot(
                            x[best_epoch],
                            best_value,
                            marker="o",
                            markersize=markersize,
                            fillstyle="full",
                            color=lines[-1].get_color(),
                        )
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_metric_keys))
        ax.set_ylabel("metric")
        fontsize = min(10, max(5, 16 - n_labels))
        ncol = min(4, 1 + n_labels // 11)
        ax.legend(fontsize=fontsize, ncol=ncol, **kwargs_legend)
        index_subplot += 1

    if len(plot_lr_keys) > 0:
        ax = axs[index_subplot]
        n_labels = 0
        for i, hist in enumerate(histories):
            for lr_name in lr_names:
                color = (
                    None
                    if simple_color_grouping
                    else lr_clrs[lr_name, model_names[i]]
                )
                _label = lr_name
                if model_names[i] != "":
                    _label += "_" + model_names[i]
                if _label is not None:
                    n_labels += 1
                lines = ax.plot(
                    x[: len(hist[lr_name])],
                    hist[lr_name],
                    label=_label,
                    color=color,
                )
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(numticks=4, subs=(0.2, 0.4, 0.6, 0.8))
        )
        ax.set_ylabel("learning rate")
        fontsize = min(10, max(5, 14 - n_labels))
        ncol = min(4, 1 + n_labels // 11)
        ax.legend(fontsize=fontsize, ncol=ncol, **kwargs_legend)
        index_subplot += 1

    axs[0].set_xlim(
        left=initial_epoch - 1, right=initial_epoch + n_epochs_max + 1
    )

    if n_subplots > 1:
        plt.subplots_adjust(hspace=0)

    return axs


def history_concatenate(prev_history: dict, last_history: dict) -> dict:
    """
    Concatenate two dictionaries in the format of the `history` attribute of
    the `History` object which is returned by the [fit](https://keras.io/api/models/model_training_apis/#fit-method)
    method of the model.

    Useful for combining histories of model fitting with two or more consecutive
    runs into a single history to plot full learning curves.

    Parameters
    ----------
    prev_history : dict
        History of the previous run of model fitting. The values of all keys
        must be lists of the same length.
    last_history : dict
        History of the last run of model fitting. The values of all keys
        must be lists of the same length.

    Returns
    -------
    dict
        Dictionary with combined histories.

    Examples
    --------
    >>> import keras
    >>> from lcurvetools import history_concatenate, lcurves_by_history

    [Create](https://keras.io/api/models/), [compile](https://keras.io/api/models/model_training_apis/#compile-method)
    and [fit](https://keras.io/api/models/model_training_apis/#fit-method) the keras model:
    >>> model = keras.Model(...) # or keras.Sequential(...)
    >>> model.compile(...)
    >>> hist1 = model.fit(...)

    Compile as needed and fit using possibly other parameter values:
    >>> model.compile(...)
    >>> hist2 = model.fit(...)

    Concatenate the `.history` dictionaries into one:
    >>> full_history = history_concatenate(hist1.history, hist2.history)

    Use `full_history` dictionary to plot full learning curves:
    >>> lcurves_by_history(full_history);
    """
    if not type(prev_history) is dict:
        raise TypeError("The `prev_history` parameter should be a dictionary.")
    if not type(last_history) is dict:
        raise TypeError("The `last_history` parameter should be a dictionary.")

    if len(prev_history) < 1:
        return last_history
    if len(last_history) < 1:
        return prev_history

    prev_epochs = set(map(len, prev_history.values()))
    if len(prev_epochs) != 1:
        raise ValueError(
            "The values of all `prev_history` keys should be lists of the same"
            " length, equaled  to the number of epochs."
        )
    prev_epochs = list(prev_epochs)[0]

    if len(set(map(len, last_history.values()))) != 1:
        raise ValueError(
            "The values of all `last_history` keys should be lists of the same"
            " length, equaled  to the number of epochs."
        )

    full_history = deepcopy(prev_history)
    for key in last_history.keys():
        if key in prev_history.keys():
            full_history[key] += last_history[key]
        else:
            full_history[key] = [None] * prev_epochs + last_history[key]

    return full_history


def lcurves_by_MLP_estimator(
    MLP_estimator: object,
    initial_epoch: int = 0,
    epoch_range_to_scale: int | list[int] | tuple[int, int] = 0,
    plot_losses: bool = True,
    plot_val_scores: bool = True,
    on_separate_subplots: bool = False,
    figsize: tuple[float, float] | None = None,
) -> list[object]:
    """
    Plot learning curves of the MLP estimator ([MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    or [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html))
    trained with the scikit-learn library as dependencies of loss and
    validation score values on the epoch index. These dependencies can be
    shown on one plot with two vertical left and right axes scaled
    independently or on two separated subplots. The best values are marked
    on the dependencies (minimum values for losses and maximum values for
    metrics).


    Parameters
    ----------
    MLP_estimator : scikit-learn estimator of `MLPClassifier` or `MLPRegressor` classes
        The estimator must be trained already using the `fit` method.

    initial_epoch : int, default=0
        The epoch index at which the `fit` method had started to train the
        model at the last run with the parameter `warm_start=True`. Also,
        setting `initial_epoch=1` can be useful to convert the epoch index
        plotted along the horizontal axes of the subplots into the number
        of passed epochs.

    epoch_range_to_scale : int or list (tuple) of int, default=0
        Specifies the epoch index range within which the vertical axes with
        loss and validation score are scaled.
        - If `epoch_range_to_scale` is a list or a tuple of two int values,
        then they specify the epoch index limits of the scaling range in the
        form `[start, stop)`, i.e. as for `slice` and `range` objects.
        - If `epoch_range_to_scale` is an int value, then it specifies the
        lower epoch index `start` of the scaling range, and the vertical axes
        are scaled by epochs with indices from `start` to the last.

        The epoch index values `start`, `stop` must take into account
        the value of the `initial_epoch` parameter.

    plot_losses : bool, default=True
        Whether to plot a dependence of loss values on epoch index.

    plot_val_scores : bool, default=True
        Whether to plot a dependence of validation score values on epoch
        index. If `MLP_estimator` doesn't have the `validation_scores_`
        attribute, the value of `plot_val_scores` is ignored and the
        dependence of validation score doesn't plot.

    on_separate_subplots : bool, default=False
        Specifies a way of showing dependences of loss and validation score
        on epoch index when `plot_losses=True`, `plot_val_scores=True` and
        `MLP_estimator` has the `validation_scores_` attribute.
        - If `True`, the dependencies are shown on two separated subplots.
        - If `False`, the dependencies are shown on one plot with two vertical
        axes scaled independently. Loss values are plotted on the left axis
        and validation score values are plotted on the right axis.

    figsize : a tuple (width, height) in inches or `None`, default=None.
        Specifies size of creating figure. If `None`, default values of width
        and height of a figure for the matplotlib library will be used.

    Returns
    -------
    numpy array or list of `matplotlib.axes.Axes` object
        - If dependencies of loss and validation score values on the epoch
        index are shown on one plot with two vertical axes scaled
        independently, the first `matplotlib.axes.Axes` object contains
        a dependence of loss values and the second `matplotlib.axes.Axes`
        object contains a dependence of validation score values.
        - If dependencies of loss and validation score values on the epoch
        index are shown on two separated subplots, each `matplotlib.axes.Axes`
        object in the numpy array or list corresponds to the built subplot
        from top to bottom.

    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from lcurvetools import lcurves_by_MLP_estimator

    [Create](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) and [fit](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.fit)
    the scikit-learn MLP estimator:
    >>> clf = MLPClassifier(..., early_stopping=True)
    >>> clf.fit(...)

    Use `clf` object with `loss_curve_` and `validation_scores_` attributes
    to plot the learning curves as the dependences of loss and validation
    score values on epoch index:
    >>> lcurves_by_MLP_estimator(clf)
    """
    from sklearn import neural_network as nn

    def get_ylims(values):
        ylim_top = max(values[epochs_slice])
        ylim_bottom = min(values[epochs_slice])
        pad = (ylim_top - ylim_bottom) * 0.05
        if pad == 0:
            pad = 0.01
        return dict(bottom=ylim_bottom - pad, top=ylim_top + pad)

    if not (
        isinstance(MLP_estimator, nn.MLPClassifier)
        or isinstance(MLP_estimator, nn.MLPRegressor)
    ):
        raise TypeError(
            "The `MLP_estimator` must be a scikit-learn MLP estimator object of"
            " `MLPClassifier` or `MLPRegressor` class."
        )
    if not hasattr(MLP_estimator, "loss_curve_"):
        raise AttributeError(
            "The `MLP_estimator` must be fitted. Run `.fit` method of the"
            " `MLP_estimator` before using `lcurves_by_MLP_estimator`."
        )
    if not (plot_losses or plot_val_scores):
        raise ValueError(
            "The value of at least one of `plot_losses` and `plot_val_scores`"
            " parameters should be `True`."
        )
    if plot_val_scores and (
        not hasattr(MLP_estimator, "validation_scores_")
        or MLP_estimator.validation_scores_ is None
    ):
        warnings.warn(
            "The `validation_scores_` attribute of the `MLP_estimator` object"
            " is not available or is `None`, so the dependence of validation"
            " score on an epoch index will not be plotted."
        )
        if not plot_losses:
            warnings.warn(
                "In addition, `plot_losses = False `, so no dependences are"
                " plotted."
            )
            return
        plot_val_scores = False

    on_separate_subplots = (
        on_separate_subplots and plot_losses and plot_val_scores
    )
    if on_separate_subplots:
        axs = lcurves_by_history(
            {
                "loss": MLP_estimator.loss_curve_,
                "validation score": MLP_estimator.validation_scores_,
            },
            initial_epoch=initial_epoch,
            epoch_range_to_scale=epoch_range_to_scale,
            plot_learning_rate=False,
            figsize=figsize,
        )
        axs[-1].set_ylabel("validation score")
        axs[0].legend().remove()
        axs[-1].legend().remove()
        return axs

    n_epochs = len(MLP_estimator.loss_curve_)

    if epoch_range_to_scale is None:
        epochs_slice = slice(0, n_epochs)
    elif type(epoch_range_to_scale) is int:
        epochs_slice = slice(
            max(0, epoch_range_to_scale - initial_epoch), n_epochs
        )
    elif (
        isinstance(epoch_range_to_scale, (list, tuple))
        and len(epoch_range_to_scale) == 2
    ):
        epochs_slice = slice(
            max(0, epoch_range_to_scale[0] - initial_epoch),
            min(n_epochs, max(1, epoch_range_to_scale[1] - initial_epoch + 1)),
        )
    else:
        raise TypeError(
            "The `epoch_range_to_scale` parameter should be an int value or a"
            " list (tuple) of two int values."
        )

    need_to_scale = 0 < epochs_slice.start or epochs_slice.stop < n_epochs

    x = range(initial_epoch, initial_epoch + n_epochs)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=figsize)

    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", right=True)
    ax.tick_params(axis="x", which="both", top=True)
    ax.set_xlabel("epoch")
    axs = [ax]
    if plot_losses and plot_val_scores:
        axs.append(plt.gca().twinx())
        axs[0].spines["right"].set_visible(False)
        axs[1].spines[["left", "top", "bottom"]].set_visible(False)
        axs[0].tick_params(axis="y", which="both", colors=colors[0])
        axs[1].minorticks_on()
        axs[1].tick_params(
            axis="y", which="both", colors=colors[1], direction="in"
        )
        axs[0].spines["left"].set_color(colors[0])
        axs[1].spines["right"].set_color(colors[1])
        axs[0].set_ylabel("loss", color=colors[0])
        axs[1].set_ylabel(
            "validation score",
            color=colors[1],
            rotation=-90,
            ha="center",
            va="bottom",
        )
        ax.grid(axis="x", linestyle="--")
        axs[0].grid(axis="y", color=colors[0], linestyle="--")
        axs[1].grid(axis="y", color=colors[1], linestyle="--")
    else:
        ax.grid()

    if plot_losses:
        axs[0].plot(x, MLP_estimator.loss_curve_, color=colors[0])
        best_value = min(MLP_estimator.loss_curve_)
        axs[0].plot(
            x[MLP_estimator.loss_curve_.index(best_value)],
            best_value,
            marker="o",
            markersize=4,
            color=colors[0],
        )
        if need_to_scale:
            axs[0].set_ylim(**get_ylims(MLP_estimator.loss_curve_))
        if not plot_val_scores:
            axs[0].set_ylabel("loss")

    if plot_val_scores:
        axs[-1].plot(x, MLP_estimator.validation_scores_, color=colors[1])
        best_value = max(MLP_estimator.validation_scores_)
        axs[-1].plot(
            x[MLP_estimator.validation_scores_.index(best_value)],
            best_value,
            marker="o",
            markersize=4,
            color=colors[1],
        )
        if need_to_scale:
            axs[-1].set_ylim(**get_ylims(MLP_estimator.validation_scores_))
        if not plot_losses:
            axs[-1].set_ylabel("validation score")

    return axs
