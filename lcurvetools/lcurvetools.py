import matplotlib.pyplot as plt
from matplotlib import ticker
import warnings


def lcurves(
    history_or_estimator,
    num_ignored_epochs=0,
    initial_epoch=0,
    plot_losses=True,
    plot_metrics=True,
    plot_learning_rate=True,
    figsize=None,
):
    """
    Plot learning curves of the neural network model trained with the keras or
    scikit-learn framework.

    Parameters
    ----------
    history_or_estimator : dictionary or scikit-learn neural network estimator
        If a dictionary, then it must contain a record of training loss values
        and metrics values at successive epochs, as well as validation loss
        values and validation metrics values (if applicable) in the format of
        the `history` attribute of the `History object`_, which is returned by
        the `fit`_ method of the model.
    num_ignored_epochs : int, default=0
        _description_
    initial_epoch : int, default=0
        _description_
    plot_losses : bool, default=True
        _description_
    plot_metrics : bool, default=True
        _description_
    plot_learning_rate : bool, default=True
        _description_

    .. _History object:
        https://keras.io/api/models/model_training_apis/#:~:text=Returns-,A%20History%20object,-.%20Its%20History.history
    .. _fit:
        https://keras.io/api/models/model_training_apis/#fit-method

    Examples
    --------
    # https://docs.python.org/3/library/doctest.html
    >>> # comments are ignored
    >>> x = 12
    >>> x
    12
    >>> if x == 13:
    ...     print("yes")
    ... else:
    ...     print("no")
    no

    # https://docs.python.org/3/library/doctest.html#doctest.ELLIPSIS
    >>> print(list(range(20)))  # doctest: +NORMALIZE_WHITESPACE
    [0,   1,  2,  3,  4,  5,  6,  7,  8,  9,
    10,  11, 12, 13, 14, 15, 16, 17, 18, 19]
    >>> print(list(range(20)))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [0,    1, ...,   18,    19]
    >>> print(list(range(20)))  # doctest: +ELLIPSIS
    ...                         # doctest: +NORMALIZE_WHITESPACE
    [0,    1, ...,   18,    19]

    Notes
    -----
    More information.  This can be in paragraph form, and uses markdown to

    - show lists
    - like this
    - with as many items as you want

    Or to show code blocks, with two colons::

        import pylab as pl
        x = np.arange(10)
        y = np.sin(x)

        pl.plot(x, y)

    We use a code block for a pylab example, because plotting does not
    play well with doctests (doctests runs all the example code, and checks
    that the output matches).
    """
    return


def lcurves_by_estimator(
    sklearn_estimator,
    num_ignored_epochs=0,
    initial_epoch=0,
    plot_losses=True,
    plot_val_scores=True,
    figsize=None,
):
    """
    Plot learning curves of the MLP estimator (classifier or regressor)
    trained with the scikit-learn library.

    Parameters
    ----------
    sklearn_estimator : scikit-learn estimator of `MLPClassifier`_ or `MLPRegressor`_ classes
        The estimator must already be trained using the `fit` method.
    num_ignored_epochs : int, default=0
        _description_
    initial_epoch : int, default=0
        _description_
    plot_losses : bool, default=True
        _description_
    plot_metrics : bool, default=True
        _description_
    plot_learning_rate : bool, default=True
        _description_

    .. _MLPClassifier:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    .. _MLPRegressor:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    """
    return


def lcurves_by_history(
    history,
    initial_epoch=0,
    epoch_range_to_scale=0,
    plot_losses=True,
    plot_metrics=True,
    plot_learning_rate=True,
    figsize=None,
):
    """
    Plots learning curves of a neural network model trained with the keras
    framework. Dependences of values of the losses, metrics and the learning
    rate on the epoch index can be plotted on three subplots along a figure
    column.

    Parameters
    ----------
    history : dict
        The dictionary could contain keys with training and validation values
        of losses and metrics, as well as learning rate values at successive
        epochs in the format of the `history` attribute of the `History`
        object which is returned by the
        [fit](https://keras.io/api/models/model_training_apis/#fit-method)
        method of the model. The values of all keys should be represented by
        numeric lists of the same length, equal to the number of epochs
        `n_epochs`.

    initial_epoch : int, default=0
        The epoch index at which the `fit` method had started to train
        the model. The parameter corresponds to the same parameter of the
        [fit](https://keras.io/api/models/model_training_apis/#fit-method)
        method of a keras model. Also, setting `initial_epoch=1` can be useful
        to convert the epoch index plotted along the horizontal axes of the
        subplots into the number of passed epochs.

    epoch_range_to_scale : int or list (tuple) of int, default=0
        Specifies the epoch index range within which the subplots of the
        losses and metrics are scaled.
        - If `epoch_range_to_scale` is a list or a tuple of two int values,
        then they specify the epoch index limits of the scaling range in the
        form `[start, stop)`, i.e. as for `slice` and `range` objects.
        - If `epoch_range_to_scale` is an int value, then it specifies the
        lower epoch index `start` of the scaling range, and the losses and
        metrics subplots are scaled by epochs with indices from `start` to the
        last.

        The epoch index values `start`, `stop` must take into account
        the value of the `initial_epoch` parameter.

    plot_losses : bool or list, default=True
        - If bool, it specifies the need to plot a subplot of losses.
        Dictionary keys with the name "loss" and names containing the
        substring "_loss" are treated as losses keys.
        - If list, it specifies loss key names of the `history` dictionary
        that should be plotted into the losses subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        `val_` of the specified key names.

    plot_metrics : bool or list, default=True
        - If bool, it specifies the need to plot a subplot of metrics.
        Dictionary keys that have not been recognized as loss or learning rate
        keys are treated as metrics keys.
        - If list, it specifies metric key names of the `history` dictionary
        that should be plotted into the metrics subplot. The subplot will also
        automatically display epoch dependencies of values with the prefix
        `val_` of the specified key names.

    plot_learning_rate : bool or list, default=True
        - If bool, it specifies the need to plot a subplot of learning rate.
        Dictionary keys with the name "lr" and names containing the
        substring "learning_rate" are treated as learning rate keys.
        - If list, it specifies learning rate key names of the `history`
        dictionary that should be plotted into the learning rate subplot.

    figsize : None or a tuple (width, height) in inches, default=None.
        Specifies size of created figure. If `None`,
        `figsize = (1.5 * default_width, 1.2 * default_height)`, where
        `default_width` and `default_height` are default width and height of
        a figure creating by matplotlib library.

    Returns
    -------
        `matplotlib.axes.Axes` or `numpy.ndarray` of them
    """

    def get_ylims(keys):
        ylim_top = -float("inf")
        ylim_bottom = float("inf")
        for key in keys:
            ylim_top = max(ylim_top, max(history[key][epochs_slice]))
            ylim_bottom = min(ylim_bottom, min(history[key][epochs_slice]))
        pad = (ylim_top - ylim_bottom) * 0.05
        if pad == 0:
            pad = 0.01
        return dict(bottom=max(0, ylim_bottom - pad), top=ylim_top + pad)

    def get_plot_keys(plot_, _keys):
        if type(plot_) is list:
            if len(plot_) > 0:
                train_keys = []
                for key_name in plot_:
                    if key_name in history.keys():
                        train_keys.append(key_name)
                    else:
                        print(
                            f"The '{key_name}' key not found in the `history` dictionary."
                        )
                return train_keys + [
                    "val_" + key_name
                    for key_name in plot_
                    if "val_" + key_name in history.keys()
                ]
        elif plot_:
            return _keys
        return []

    if not type(history) is dict:
        raise TypeError("The `history` parameter should be a dictionary.")
    if len(history) < 1:
        raise ValueError("The `history` dictionary cannot be empty.")
    set_lengths = set(map(len, history.values()))
    if len(set_lengths) != 1:
        raise TypeError(
            "The values of all `history` keys should be lists of the same length, equal to the number of epochs."
        )
    n_epochs = list(set_lengths)[0]

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
            "The `epoch_range_to_scale` parameter should be an int value or a list (or a tuple) of two int values."
        )

    if type(plot_losses) not in [bool, list, tuple]:
        raise TypeError("Параметр plot_losses повинен мати тип bool або list")
    if type(plot_metrics) not in [bool, list, tuple]:
        raise TypeError("Параметр plot_metrics повинен мати тип bool або list")
    if type(plot_learning_rate) not in [bool, list, tuple]:
        raise TypeError(
            "Параметр plot_learning_rate повинен мати тип bool або list"
        )

    # бажана перевірка, щоб не було повторів параметрів на різних графіках

    loss_keys = [
        name for name in history.keys() if name == "loss" or "_loss" in name
    ]
    lr_keys = [
        name
        for name in history.keys()
        if "lr" == name or "learning_rate" in name
    ]
    metric_keys = [
        name for name in history.keys() if name not in (loss_keys + lr_keys)
    ]

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
    # бажана перевірка, щоб не було повторів параметрів на різних графіках

    # n_epochs = len(history["loss"])
    need_to_scale = 0 < epochs_slice.start or epochs_slice.stop < n_epochs

    fig = plt.figure()  # plt.gcf()
    if n_subplots > 1:
        if n_subplots == 2:
            axs = fig.subplots(n_subplots, 1, sharex=True)
        else:
            axs = fig.subplots(
                n_subplots, 1, sharex=True, height_ratios=[2, 2, 1]
            )
    else:
        axs = [plt.gca()]

    # if len(plot_lr_keys) > 0:
    #     axs[-1].set_yscale("log", base=10)

    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(
            axis="x",
            which="minor",
            direction="in",
            # length=3,
            # direction="inout",
            # length=5,
            bottom=True,
            top=True,
        )
        ax.tick_params(
            axis="x",
            which="major",
            direction="in",
            # length=5,
            # direction="inout",
            # length=7,
            bottom=True,
            top=True,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            direction="in",
            # length=5,
            left=True,
            labelleft=True,
            right=True,
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="in",
            # length=7,
            left=True,
            labelleft=True,
            right=True,
        )
        ax.yaxis.set_label_position("left")
        ax.grid()
        # ax.grid(which="both")

    # axs[0].tick_params(axis="x", labeltop=True)
    axs[-1].tick_params(axis="x", labelbottom=True)
    axs[-1].set_xlabel("epoch")

    x = range(initial_epoch, initial_epoch + n_epochs)

    index_subplot = 0
    kwargs_legend = dict(loc="upper left", bbox_to_anchor=(1.002, 1))

    if len(plot_loss_keys) > 0:
        ax = axs[index_subplot]
        for key in plot_loss_keys:
            ax.plot(x, history[key])
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_loss_keys))
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.set_ylabel("losses")
        ax.legend(plot_loss_keys, **kwargs_legend)
        index_subplot += 1

    if len(plot_metric_keys) > 0:
        ax = axs[index_subplot]
        for key in plot_metric_keys:
            ax.plot(x, history[key])
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_metric_keys))
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.set_ylabel("metrics")
        ax.legend(plot_metric_keys, **kwargs_legend)
        index_subplot += 1

    if len(plot_lr_keys) > 0:
        ax = axs[index_subplot]
        for key in plot_lr_keys:
            ax.plot(x, history[key])
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(numticks=4, subs=(0.2, 0.4, 0.6, 0.8))
        )
        # if need_to_scale:
        #     ax.set_ylim(**get_ylims(plot_lr_keys))

        ax.set_ylabel("learning rate")
        ax.legend(plot_lr_keys, **kwargs_legend)
        index_subplot += 1

    axs[0].set_xlim(left=initial_epoch)
    figwidth = fig.get_figwidth() * 1.5
    figheight = fig.get_figheight() * 1.2
    if n_subplots > 1:
        plt.subplots_adjust(hspace=0)

    if figsize is None:
        # it prints no text "<Figure size ...>"
        fig.set_size_inches(figwidth, figheight)
    else:
        fig.set_size_inches(figsize)

    if len(axs) > 1:
        return axs
    return axs[0]


def history_concatenate(history, last_history):
    full_history = history.copy()
    for key in last_history.keys():
        if key in full_history.keys():
            full_history[key] += last_history[key]
        else:
            full_history[key] = last_history[key]
    return full_history
