import matplotlib.pyplot as plt
from matplotlib import ticker
from copy import deepcopy
import pandas as pd
import warnings

from .utils import get_mode_by_metric_name, get_best_epoch_value


def _get_n_epochs(history):
    n_epochs = set(map(len, history.values()))
    if len(n_epochs) != 1:
        raise TypeError(
            "The values of all `history` keys should be lists of the same"
            " length, equaled to the number of epochs."
        )
    return list(n_epochs)[0]


def _get_train_metric_names(keys):
    train_keys = []
    for key in keys:
        if key.startswith("val_"):
            train_keys.append(key[4:])
        else:
            train_keys.append(key)
    return list(set(train_keys))


def _get_colors(
    num: int, paired_colors: bool
) -> tuple[tuple[float, float, float]]:
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    tab20_cmap = plt.get_cmap("tab20").colors
    tab20b_cmap = (
        plt.get_cmap("tab20b").colors[::2] + plt.get_cmap("tab20b").colors[1::2]
    )
    tab20c_cmap = (
        plt.get_cmap("tab20c").colors[::2] + plt.get_cmap("tab20c").colors[1::2]
    )
    cmap = tab20_cmap + tab20b_cmap + tab20c_cmap
    if not paired_colors:
        cmap = cmap[::2] + cmap[1::2]
    while len(cmap) < num:
        cmap += cmap
    return cmap[:num]


def lcurves_by_history(
    history: dict | list[dict],
    initial_epoch: int = 0,
    epoch_range_to_scale: int | list[int] | tuple[int, int] = 0,
    plot_losses: bool | list[str] = True,
    plot_metrics: bool | list[str] = True,
    plot_learning_rate: bool | list[str] = True,
    unique_curve_colors: bool = False,  # слід переробити на color_groupping_by = 'auto' | 'model' | 'subset' | None ("auto" - для однієї моделі "subset", для кількох - "model", "model" - кожна модель своїм кольором, "subset" - кожен піднабір кривих своїм кольором, None - всі криві однаковим кольором)
    color_grouping_by: str = "auto",
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
    history : dict or list of dict
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
        - If list, it specifies loss key names of the `history` dictionary
        that should be plotted into the losses subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        'val_' of the specified key names.

    plot_metrics : bool or list of str, default=True
        - If bool, it specifies the need to plot a subplot of metrics.
        Dictionary keys that have not been recognized as loss or learning rate
        keys are treated as metrics keys.
        - If list, it specifies metric key names of the `history` dictionary
        that should be plotted into the metrics subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        'val_' of the specified key names.

    plot_learning_rate : bool or list of str, default=True
        - If bool, it specifies the need to plot a subplot of learning rate.
        Dictionary keys with the name "lr" and names containing the
        substring "learning_rate" are treated as learning rate keys.
        - If list, it specifies learning rate key names of the `history`
        dictionary that should be plotted into the learning rate subplot.

        Learning rate values on the vertical axis are plotted in a logarithmic
        scale.

    unique_curve_colors : bool, default=False
        If True, each curve inside the subplots will be assigned a unique color.
        If False, curves for each metric will share the same color.

    model_names : list of str or None, default=None
        Specifies model names for each history in the `history` list. The names
        will be used in the legends of the subplots if `unique_curve_colors` is
        True. The length of the `model_names` list must be equal to the length
        of the `history` list.
        If `None`, the function will use default names.
        If `history` is a single dictionary or a list with a single dictionary
        or `unique_curve_colors=False`, the `model_names`parameter is ignored.

    optimization_modes : dict or None, default=None
        Specifies optimization modes for each metric name in the `history` dictionary.
        For example, if `optimization_modes = {"iou": "max", "hinge": "min"}`
        the function will consider iou as metric for maximization
        and hinge as metric for minimization. If a metric name is not
        found in the `optimization_modes` dictionary, the metric mode will be
        determined automatically with the function `lcurvetools.utils.get_optimization_mode`.
        If `None`, the "auto" mode will be used for all metrics.
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
        for hist in history:
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
        if type(plot_) is list:
            if len(plot_) > 0:
                keys = []
                for hist in history:
                    for key_name in plot_:
                        if key_name in hist.keys():
                            keys.append(key_name)
                    keys += [
                        "val_" + key_name
                        for key_name in plot_
                        if "val_" + key_name in hist.keys()
                    ]
                return sorted(list(set(keys)))
        elif plot_:
            return sorted(_keys)
        return []

    # Input data validation

    if not isinstance(history, (list, dict)):
        raise TypeError(
            "The `history` parameter should be a dictionary or a list of"
            " dictionaries."
        )
    if len(history) == 0:
        raise ValueError("The `history` list or dictionary cannot be empty.")
    if isinstance(history, list):
        for i, hist in enumerate(history):
            if not type(hist) is dict:
                raise TypeError(
                    f"The {i}-th element of the `history` list is not a"
                    " dictionary."
                )
            if len(hist) == 0:
                raise ValueError(
                    f"The {i}-th dictionary in the `history` list cannot be"
                    " empty."
                )
    if type(history) is dict:
        history = [history]
    n_epochs = [_get_n_epochs(hist) for hist in history]
    n_epochs_max = max(n_epochs)

    if epoch_range_to_scale is None:
        epochs_slice = slice(0, n_epochs_max)
    elif type(epoch_range_to_scale) is int:
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
            "The `epoch_range_to_scale` parameter should be an int value or a"
            " list (tuple) of two int values."
        )

    if type(plot_losses) not in [bool, list, tuple]:
        raise TypeError(
            "The `plot_losses` parameter should be bool, list or tuple"
        )
    if type(plot_metrics) not in [bool, list, tuple]:
        raise TypeError(
            "The `plot_metrics` parameter should be bool, list or tuple"
        )
    if type(plot_learning_rate) not in [bool, list, tuple]:
        raise TypeError(
            "The `plot_learning_rate` parameter should be bool, list or tuple"
        )

    if optimization_modes is not None:
        if not isinstance(optimization_modes, dict):
            raise TypeError(
                "The `optimization_modes` parameter should be a dictionary or"
                " None."
            )
        for key, value in optimization_modes.items():
            if not isinstance(key, str) or len(key) == 0:
                raise TypeError(
                    "The keys of the `optimization_modes` dictionary should be"
                    " non-empty strings."
                )
            if value not in ("min", "max"):
                raise ValueError(
                    "The values of the `optimization_modes` dictionary should"
                    " be 'min' or 'max'."
                )
    if not isinstance(initial_epoch, int) or initial_epoch < 0:
        raise ValueError(
            "The `initial_epoch` parameter should be a non-negative integer."
        )
    if not isinstance(unique_curve_colors, bool):
        raise TypeError(
            "The `unique_curve_colors` parameter should be a boolean value."
        )
    # if len(history) > 1 and unique_curve_colors:
    if model_names is not None:
        if not isinstance(model_names, list):
            raise TypeError(
                "The `model_names` parameter should be a list of strings or"
                " None."
            )
        if len(model_names) != len(history):
            raise ValueError(
                "The length of the `model_names` list should be equal to"
                " the length of the `history` list."
            )
        for name in model_names:
            if not isinstance(name, str):
                raise TypeError(
                    "The elements of the `model_names` list should be strings."
                )
    else:
        model_names = list(str(i) for i in range(len(history)))
    if color_grouping_by not in ("auto", "model", "subset", None):
        raise ValueError(
            "The `color_grouping_by` parameter should be 'auto', 'model',"
            " 'subset' or None."
        )
    if color_grouping_by == "auto":
        if len(history) > 1:
            color_grouping_by = "model"
        else:
            color_grouping_by = "subset"
    # End of input data validation

    # Extract keys for losses, learning rates, and metrics
    loss_keys = []
    lr_keys = []
    metric_keys = []
    for hist in history:
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

    train_loss_names = _get_train_metric_names(plot_loss_keys)
    train_metric_names = _get_train_metric_names(plot_metric_keys)

    val_key_exists = False
    for key in plot_loss_keys + plot_metric_keys:
        if key[:4] == "val_":
            val_key_exists = True
            break

    # determine colors of curves in the subplots of losses, metrics and learning rates
    prefix_index = [""]
    if val_key_exists:
        prefix_index.append("val_")

    index = pd.MultiIndex.from_product(
        [model_names, prefix_index], names=["model", "prefix"]
    )
    loss_clrs = pd.Series(
        _get_colors(len(index), paired_colors=val_key_exists), index=index
    )

    index = pd.MultiIndex.from_product(
        [model_names, train_metric_names, prefix_index],
        names=["model", "metric", "prefix"],
    )
    metric_clrs = pd.Series(
        _get_colors(len(index), paired_colors=val_key_exists), index=index
    )

    index = pd.MultiIndex.from_product(
        [model_names, plot_lr_keys], names=["model", "lr"]
    )
    lr_clrs = pd.Series(
        _get_colors(len(index), paired_colors=False), index=index
    )

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
        for i, hist in enumerate(history):
            for key in train_loss_names:
                for prefix in prefix_index:
                    _key = prefix + key
                    if _key in hist.keys():
                        color = loss_clrs[model_names[i], prefix]
                        _label = _key + "_" + model_names[i]
                        if _label is not None:
                            n_labels += 1
                        lines = ax.plot(
                            x[: len(hist[_key])],
                            hist[_key],
                            label=_label,
                            color=color,
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
        # for key in plot_loss_keys:
        #     # color = None
        #     label = key
        #     for i, hist in enumerate(history):
        #         if key not in hist.keys():
        #             continue
        #         color = loss_clrs[
        #             model_names[i], "val" if key[:4] == "val_" else ""
        #         ]
        #         # _label = (
        #         #     label + f"_{model_names[i]}"
        #         #     if unique_curve_colors
        #         #     else label
        #         # )
        #         _label = key + "_" + model_names[i]
        #         if _label is not None:
        #             n_labels += 1
        #         lines = ax.plot(
        #             x[: len(hist[key])],
        #             hist[key],
        #             label=_label,
        #             color=color,
        #             alpha=alpha,
        #         )
        #         if not unique_curve_colors:
        #             if label is not None:
        #                 label = None
        #             if color is None:
        #                 color = lines[-1].get_color()
        #         best_epoch, best_value = get_best_epoch_value(
        #             hist[key], key, mode="min", verbose=False
        #         )
        #         ax.plot(
        #             x[best_epoch],
        #             best_value,
        #             marker="o",
        #             markersize=markersize,
        #             fillstyle="full",
        #             color=lines[-1].get_color(),
        #             alpha=alpha,
        #         )
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_loss_keys))
        ax.set_ylabel("loss")
        fontsize = min(
            10, max(5, 16 - n_labels)
        )  # if unique_curve_colors else 10
        ax.legend(fontsize=fontsize, **kwargs_legend)
        index_subplot += 1

    if len(plot_metric_keys) > 0:
        ax = axs[index_subplot]
        n_labels = 0
        for i, hist in enumerate(history):
            for key in train_metric_names:
                for prefix in prefix_index:
                    _key = prefix + key
                    if _key in hist.keys():
                        color = metric_clrs[model_names[i], key, prefix]
                        _label = _key + "_" + model_names[i]
                        if _label is not None:
                            n_labels += 1
                        lines = ax.plot(
                            x[: len(hist[_key])],
                            hist[_key],
                            label=_label,
                            color=color,
                        )
                        mode = (
                            optimization_modes[key]
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
        # for key in plot_metric_keys:
        #     color = None
        #     label = key
        #     for i, hist in enumerate(history):
        #         if key not in hist.keys():
        #             continue
        #         _label = (
        #             label + f"_{model_names[i]}"
        #             if unique_curve_colors
        #             else label
        #         )
        #         if _label is not None:
        #             n_labels += 1
        #         lines = ax.plot(
        #             x[: len(hist[key])],
        #             hist[key],
        #             label=_label,
        #             color=color,
        #             # color=plt.get_cmap("tab10", 0),
        #         )
        #         if not unique_curve_colors:
        #             if label is not None:
        #                 label = None
        #             if color is None:
        #                 color = lines[-1].get_color()
        #         mode = optimization_modes[key] if optimization_modes else "auto"
        #         best_epoch, best_value = get_best_epoch_value(
        #             hist[key], key, mode=mode, verbose=False
        #         )
        #         ax.plot(
        #             x[best_epoch],
        #             best_value,
        #             marker="o",
        #             markersize=markersize,
        #             fillstyle="full",
        #             color=lines[-1].get_color(),
        #         )
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_metric_keys))
        ax.set_ylabel("metric")
        fontsize = min(
            10, max(5, 16 - n_labels)
        )  # if unique_curve_colors else 10
        ax.legend(fontsize=fontsize, **kwargs_legend)
        index_subplot += 1

    if len(plot_lr_keys) > 0:
        ax = axs[index_subplot]
        n_labels = 0
        for i, hist in enumerate(history):
            for key in plot_lr_keys:
                color = lr_clrs[model_names[i], key]
                _label = (
                    key.replace("learning_rate", "lr") + "_" + model_names[i]
                )
                if _label is not None:
                    n_labels += 1
                lines = ax.plot(
                    x[: len(hist[key])],
                    hist[key],
                    label=_label,
                    color=color,
                    # alpha=0.7,
                )
        # for key in plot_lr_keys:
        #     color = None
        #     label = key
        #     for i, hist in enumerate(history):
        #         if key not in hist.keys():
        #             continue
        #         _label = (
        #             label + f"_{model_names[i]}"
        #             if unique_curve_colors
        #             else label
        #         )
        #         if _label is not None:
        #             n_labels += 1
        #         lines = ax.plot(
        #             x[: len(hist[key])],
        #             hist[key],
        #             label=_label,
        #             color=color,
        #         )
        #         if not unique_curve_colors:
        #             if label is not None:
        #                 label = None
        #             if color is None:
        #                 color = lines[-1].get_color()
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(numticks=4, subs=(0.2, 0.4, 0.6, 0.8))
        )
        ax.set_ylabel("learning rate")
        fontsize = min(
            10, max(5, 14 - n_labels)
        )  # if unique_curve_colors else 10
        ax.legend(fontsize=fontsize, **kwargs_legend)
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
