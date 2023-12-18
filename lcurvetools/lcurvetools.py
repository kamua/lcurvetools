import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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
    Plot learning curves of the neural network model created with the keras or
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
    num_ignored_epochs=0,
    initial_epoch=0,
    plot_losses=True,
    plot_metrics=True,
    plot_learning_rate=True,
    figsize=None,
):
    """
    Plot learning curves of a neural network model trained with the keras
    framework on three subplots with dependences of values of the loss
    functions, metrics and the learning rate on the epoch index.

    Parameters
    ----------
    history : dict
        The dictionary should be a record of training loss values
        and metrics values, validation loss values and validation metrics
        values (if applicable), as well as learning rate values (if applicable)
        at successive epochs in the format of the `history` attribute of the
        [History object](https://keras.io/api/models/model_training_apis/#:~:text=Returns-,A%20History%20object,-.%20Its%20History.history),
        which is returned by the [fit](https://keras.io/api/models/model_training_apis/#fit-method)
        method of the model. At least one of the dictionary keys should be
        `loss`.

    num_ignored_epochs : int, default=0
        The number of initial epochs that are ignored when fitting the limits
        of the vertical axes of the plots. If `num_ignored_epochs` is outside
        the range `(initial_epoch, initial_epoch + n_epochs)` where `n_epochs`
        is a number of epochs represented in the `history`, then
        the limits are fitted over all epochs.

    initial_epoch : int, default=0
        The epoch at which the `fit` method had started to train the model.
        The parameter corresponds to the [parameter with the same name](https://keras.io/api/models/model_training_apis/#fit-method:~:text=in%20compile()%20instead.-,initial_epoch,-%3A%20Integer.%20Epoch%20at)
        of the `fit` method. Also, setting `initial_epoch=1` can be useful to
        convert the epoch index labeled at the horizontal axes of the plots
        into the number of passed epochs.

    plot_losses : bool, default=True
        _description_
    plot_metrics : bool, default=True
        _description_
    plot_learning_rate : bool, default=True
        _description_

    """

    def get_ylims(keys):
        ylim_top = -float("inf")
        ylim_bottom = float("inf")
        first_epoch_index = max(0, num_ignored_epochs - initial_epoch)
        for key in keys:
            ylim_top = max(ylim_top, max(history[key][first_epoch_index:]))
            ylim_bottom = min(ylim_bottom, min(history[key][first_epoch_index:]))
        pad = (ylim_top - ylim_bottom) * 0.05
        if pad == 0:
            pad = 0.01
        return dict(bottom=max(0, ylim_bottom - pad), top=ylim_top + pad)

    def get_plot_keys(plot_, _keys):
        if type(plot_) is list:
            if len(plot_) > 0:
                train_keys = []
                for key_name in plot_:
                    if key_name in history.keys():  # _keys:
                        train_keys.append(key_name)
                    else:
                        print("Не знайдено ключ " + key_name + " в словнику history")
                return train_keys + [
                    "val_" + key_name
                    for key_name in plot_
                    if "val_" + key_name in history.keys()
                ]
        elif plot_:
            return _keys
        return []

    if not type(history) is dict:
        raise TypeError("history має бути словником")
    if "loss" not in history.keys() < 1:
        raise TypeError("Словник history повинен мати ключ 'loss'")
    if type(plot_losses) not in [bool, list]:
        raise TypeError("Параметр plot_losses повинен мати тип bool або list")
    if type(plot_metrics) not in [bool, list]:
        raise TypeError("Параметр plot_metrics повинен мати тип bool або list")
    if type(plot_learning_rate) not in [bool, list]:
        raise TypeError("Параметр plot_learning_rate повинен мати тип bool або list")
    # бажана перевірка, щоб не було повторів параметрів на різних графіках

    loss_keys = [name for name in history.keys() if name == "loss" or "_loss" in name]
    lr_keys = [name for name in history.keys() if "learning_rate" in name]
    metric_keys = [name for name in history.keys() if name not in (loss_keys + lr_keys)]

    plot_loss_keys = get_plot_keys(plot_losses, loss_keys)
    n_subplots = int(len(plot_loss_keys) > 0)

    metric_keys = [key for key in metric_keys if key not in plot_loss_keys]
    plot_metric_keys = get_plot_keys(plot_metrics, metric_keys)
    n_subplots += int(len(plot_metric_keys) > 0)

    lr_keys = [key for key in lr_keys if key not in plot_loss_keys + plot_metric_keys]
    plot_lr_keys = get_plot_keys(plot_learning_rate, lr_keys)
    n_subplots += int(len(plot_lr_keys) > 0)
    # бажана перевірка, щоб не було повторів параметрів на різних графіках

    n_epochs = len(history["loss"])
    need_to_scale = (
        initial_epoch < num_ignored_epochs
        and num_ignored_epochs < initial_epoch + n_epochs
    )

    fig = plt.gcf()
    if n_subplots > 1:
        if n_subplots == 2:
            axs = fig.subplots(n_subplots, 1, sharex=True)
        else:
            axs = fig.subplots(n_subplots, 1, sharex=True, height_ratios=[2, 2, 1])
        # plt.subplot(n_subplots, 1, 1).xaxis.set_label_position("top")
    else:
        axs = [plt.gca()]

    for i, ax in enumerate(axs):
        # ax.tick_params(bottom=False, labelbottom=False, labelright=True, right=True)
        ax.minorticks_on()
        ax.tick_params(
            axis="x", which="minor", direction="inout", length=5, bottom=True, top=True
        )
        ax.tick_params(
            axis="x", which="major", direction="inout", length=7, bottom=True, top=True
        )
        # ax.tick_params(axis="x", which="major", direction='in', top=True)
        # ax.tick_params(axis="x", which="minor", direction='inout', length=4, bottom=True, top=True)
        # ax.tick_params(axis="y", which="both", direction='out', left=True, labelleft=True, right=True, labelright=True)
        ax.tick_params(
            axis="y",
            which="minor",
            direction="inout",
            length=5,
            left=True,
            labelleft=True,
            right=True,
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="inout",
            length=7,
            left=True,
            labelleft=True,
            right=True,
        )
        # ax.tick_params(axis="y", which="both", direction='in', right=True)
        # ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position("left")
        # ax.tick_params(axis="x", which="minor", direction="out",
        #               top=True, labeltop=True, bottom=False, labelbottom=False)

        ax.grid()
    axs[0].tick_params(axis="x", labeltop=True)
    # axs[0].set_xlabel("epoch")
    # axs[0].xaxis.set_label_position('top')
    axs[-1].tick_params(axis="x", labelbottom=True)
    axs[-1].set_xlabel("epoch")
    # axs[-1].tick_params(axis="x", which="both", direction='out', bottom=True, labelbottom=True)

    x = range(initial_epoch, initial_epoch + n_epochs)

    index_subplot = 0
    kwargs_legend = dict(loc="upper left", bbox_to_anchor=(1.01, 1))

    if len(plot_loss_keys) > 0:
        ax = axs[index_subplot]
        for key in plot_loss_keys:
            ax.plot(x, history[key])
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_loss_keys))

        ax.set_ylabel("losses")
        # plt.xlabel("epoch")
        ax.legend(plot_loss_keys, **kwargs_legend)
        # plt.grid()
        # plt.tick_params(which='minor', length=5)
        index_subplot += 1

    if len(plot_metric_keys) > 0:
        # if n_subplots > 1:
        #     if n_subplots == 3:
        #         plt.subplot(n_subplots, 1, 2)
        #     else:
        #         if len(plot_loss_keys) > 0:
        #             plt.subplot(n_subplots, 1, 2)
        #         else:
        #             plt.subplot(n_subplots, 1, 1)
        ax = axs[index_subplot]
        for key in plot_metric_keys:
            ax.plot(x, history[key])
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_metric_keys))

        ax.set_ylabel("metrics")
        # plt.xlabel("epoch")
        ax.legend(plot_metric_keys, **kwargs_legend)
        # plt.grid()
        index_subplot += 1

    if len(plot_lr_keys) > 0:
        # if n_subplots > 1:
        #     plt.subplot(n_subplots, 1, n_subplots)
        ax = axs[index_subplot]
        for key in plot_lr_keys:
            ax.plot(x, history[key])
        if need_to_scale:
            ax.set_ylim(**get_ylims(plot_lr_keys))

        ax.set_ylabel("learning rate")
        # plt.xlabel("epoch")
        ax.legend(plot_lr_keys, **kwargs_legend)
        # plt.grid()
        ax.set_yscalee("log", base=10)
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
    plt.show()
    return
