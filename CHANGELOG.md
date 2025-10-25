# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2025-10-20

### Added

- Added the ability to plot multiple fitting histories returned by the [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) function on a single figure with the `lcurves_by_history` function.
- Added such parameters for the `lcurves_by_history` function:
  - `color_grouping_by` which specifies how colors of curves in subplots are grouped;
  - `model_names` which specifies model names for each history in the `history` list to use in the legends of the subplots;
  - `optimization_modes` which specifies optimization modes of metrics ("min" or "max").
- Added the `utils` module with two functions: `get_mode_by_metric_name` and `get_best_epoch_value`.

### Changed

- The `history` parameter of the `lcurves_by_history` function has been renamed to `histories` and can now accept a list of dictionaries with several fitting histories of keras models.

## [1.0.1] - 2025-01-23

### Changed

The default value for the `figsize` parameter of the `lcurves_by_history` and `lcurves_by_MLP_estimator` functions has been changed to `None`.

## [1.0.0] - 2024-01-22

Initial release.
