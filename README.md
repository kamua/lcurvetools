# lcurvetools

Simple tools to plot learning curves of a neural network model trained with the keras or scikit-learn framework.

## The `lcurves_by_history` function to plot learning curves by the `history` attribute of the keras `History` object

Neural network model training with keras is performed using the [fit](https://keras.io/api/models/model_training_apis/#fit-method) method. The method returns the `History` object with the `history` attribute which is dictionary and contains keys with training and validation values of losses and metrics, as well as learning rate values at successive epochs. The `lcurves_by_history` function uses the `History.history` dictionary to plot the learning curves as the dependences of the above values on the epoch index.

### Usage scheme

- Import the `keras` module and the `lcurves_by_history` function:

```python
import keras
from lcurvetools import lcurves_by_history
```

- [Create](https://keras.io/api/models/), [compile](https://keras.io/api/models/model_training_apis/#compile-method)
and [fit](https://keras.io/api/models/model_training_apis/#fit-method) the keras model:

```python
model = keras.Model(...) # or keras.Sequential(...)
model.compile(...)
hist = model.fit(...)
```

- Use `hist.history` dictionary to plot the learning curves as the dependences of values of all keys in the dictionary on an epoch index with automatic recognition of keys of losses, metrics and learning rate:

```python
lcurves_by_history(hist.history)
```

## Typical appearances of the output figure

The appearance of the output figure depends on the list of keys in the `hist.history` dictionary, which is determined by the parameters of the `compile` and `fit` methods of the model. For example, for a typical usage of these methods, the list of keys would be `['loss', 'accuracy', 'val_loss', 'val_accuracy']` and the output figure will contain 2 subplots with loss and metrics vertical axes and might look like this:

```python
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=50)
lcurves_by_history(hist.history);
```

![typical plot of learning curves](img/typical_plot.png)

**Note:** the best values are marked for dependencies of losses and metrics (minimum values for losses and maximum values for metrics).

Of course, if the `metrics` parameter of the `compile` method is not specified, then the output figure will not contain a metric subplot.

Usage of callbacks for the `fit` method can add new keys to the `hist.history` dictionary. For example, the [ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) callback adds the `lr` key with learning rate values for successive epochs. In this case the output figure will contain additional subplot with learning rate vertical axis in a logarithmic scale and might look like this:

```python
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=50,
    callbacks=keras.callbacks.ReduceLROnPlateau(),
)
lcurves_by_history(hist.history);
```

![figure with learning rate subplot](img/learning_rate_subplot.png)

The `lcurves_by_history` function has optional parameters to customize the appearance of the output figure. For example, the `epoch_range_to_scale` option allows to specify the epoch index range within which the subplots of the losses and metrics are scaled.

- If `epoch_range_to_scale` is a list or a tuple of two int values, then they specify the epoch index limits of the scaling range in the form `[start, stop)`, i.e. as for `slice` and `range` objects.
- If `epoch_range_to_scale` is an int value, then it specifies the lower epoch index `start` of the scaling range, and the losses and metrics subplots are scaled by epochs with indices from `start` to the last.

So, you can exclude the first 5 epochs from the scaling range as follows:

```python
lcurves_by_history(hist.history, epoch_range_to_scale=5);
```

![figure with custom scaling](img/custom_scaling.png)
