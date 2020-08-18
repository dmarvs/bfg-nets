import datetime
import logging
import os
from typing import List
import numpy as np

import keras
import keras.backend as K

from bfgn.configuration import configs
from bfgn.experiments import experiments, histories

#from keras_one_cycle_clr import CLR

_logger = logging.getLogger(__name__)


_DIR_TENSORBOARD = "tensorboard"


class HistoryCheckpoint(keras.callbacks.Callback):
    """
    A custom Keras callback for checkpointing model training history and associated information.
    """

    config = None
    existing_history = None
    period = None
    verbose = None
    epochs_since_last_save = None
    epoch_begin = None

    def __init__(self, config: configs.Config, existing_history=None, period=1, verbose=0):
        super().__init__()
        self.config = config
        if existing_history is None:
            existing_history = dict()
        self.existing_history = existing_history
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.epoch_begin = None

    def on_train_begin(self, logs=None):
        _logger.debug("Beginning network training")
        for key in ("epoch_start", "epoch_finish"):
            self.existing_history.setdefault(key, list())
        self.existing_history["train_start"] = datetime.datetime.now()
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        _logger.debug("Ending network training")
        self.existing_history["train_finish"] = datetime.datetime.now()
        self._save_history()

    def on_epoch_begin(self, epoch, logs=None):
        _logger.debug("Beginning new epoch")
        self.epoch_begin = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        _logger.debug("Ending epoch")
        # Update times
        epoch_end = datetime.datetime.now()
        self.existing_history["epoch_start"].append(self.epoch_begin)
        self.existing_history["epoch_finish"].append(epoch_end)
        self.epoch_begin = None
        # Save if necessary
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            _logger.debug("Checkpointing model")
            self._save_history()
            self.epochs_since_last_save = 0

    def _save_history(self):
        _logger.debug("Save model history")
        if hasattr(self.model, "history"):
            new_history = self.model.history.history
        elif hasattr(self.model, "model"):
            assert hasattr(
                self.model.model, "history"
            ), "Parallel models are doing something unusual with histories. Tell Nick and let's debug."
            new_history = self.model.model.history
        combined_history = histories.combine_histories(self.existing_history, new_history)
        histories.save_history(combined_history, experiments.get_history_filepath(self.config))


#class CyclicalLearning(CLR):
#
#    def __init__(self, config: configs.Config, cycles, learning, verbose, amplitude):
#        self.config = config
#        super().__init__(
#            cyc = cycles,
#            lr_range = (learning[0], learning[1]),
#            momentum_range = (0.95, 0.85),
#            amplitude_fn = lambda x: np.power(self.amplitude, x),
#            verbose = verbose,
#            batch_size = None, # self.config.data_samples.batch_size,
#            batches_per_epoch = self.config.model_training.batches_per_epoch,
#        )
#        print("Cyclical Learning is ON with keys")


class CustomMetric(keras.callbacks.Callback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.targest = []
        self.outputs = []

        self.var_y_true = K.Variable(0., validate_shape=False)
        self.var_y_pred = K.Varibale(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))

        rmse = K.pow(K.mean(K.pow(targets - outputs, 2)), 0.5)
        mae  = K.mean(K.abs(targets - outputs))
        print(" - val_rmse: {:f} - val_mae: {:f}".format(rmse, mae))
        return

class PrintValMetrics(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._get_pred = None
        self._get_true = None
        self.preds = []
        self.true  = []

    def _metric_callback(self, preds, true):
        self.preds.append(preds)
        self.true.append(true)
        
        rmse = K.pow(K.mean(K.pow(self.true - self.preds, 2)), 0.5)
        mae  = K.mean(K.abs(self.true - self.preds))

        print(" - val_rmse: {:f} - val_mae {:f}".format(rmse, mae))


    def set_model(self, model):
        super().set_model(model)
        if self._get_pred is None:
            self._get_pred = self.model.outputs[0]
        if self._get_true is None:
            self._get_true = self.model.targets[0]

    def on_test_begin(self, logs):
        self.model._make_test_function()
        if self._get_pred not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self._get_pred)
            self.model.test_function.fetch_callbacks[self._get_pred] = self._pred_callback
        if self._get_true not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self._get_true)
            self.model.test_function.fetch_callbacks[self._get_true] = self._true_callback

    def on_test_end(self, logs):
        if self._get_pred in self.model.test_function.fetches:
            self.model.test_function.fetches.remove(self._get_pred)
        if self._get_pred in self.model.test_function.fetch_callbacks:
            self.model.test_function.fetch_callbacks.pop(self._get_pred)
        
        if self._get_true in self.model.test_function.fetches:
            self.model.test_function.fetches.remove(self._get_true)
        if self._get_true in self.model.test_function.fetch_callbacks:
            self.model.test_function.fetch_callbacks.pop(self._get_true)



class CustomMetrics(keras.callbacks.Callback):
    """
    A custom Keras callback for returning custom metrics without transformations applied.
    """
    
    val_sequence = None
    raw_features = None
    raw_responses = None
    trans_features = None
    trans_responses = None
    weights = None
    num_samples = None
    num_features = None
    num_responses = None
    trans_predictions = None
    raw_predictions = None

    def __init__(self, validation_sequence):
        self.validation_sequence = validation_sequence

    def on_epoch_end(self):
        
        self.scores = {
                'rmse' : [],
                'mae'  : [],
        }

        try:
            self.val_sequence = self.data_container.validation_sequence
            self._get_sampled_features_responses_and_set_metadata_and_weights(0)
            self.trans_predictions = model.predict(self.trans_features)
            self.raw_predictions = self.data_sequence.response_scaler.inverse_transform(self.trans_predictions)
        except:
            import ipdb;ipdb.set_trace()


    def get_sampled_features_responses_and_set_metadata_and_weights(self) -> None:
        (raw_features, raw_responses), (
            trans_features,
            trans_responses,
        ) = self.data_sequence.get_raw_and_transformed_sample(0)
        # We expect weights to be the last element in the responses array
        self.weights = trans_responses[0][..., -1]
        # Unpack features and responses, inverse transform to get raw values
        self.raw_features = raw_features[0]
        self.trans_features = trans_features[0]
        self.raw_responses = raw_responses[0][..., :-1]
        self.trans_responses = trans_responses[0][..., :-1]
        # Set sample metadata
        self.num_samples = self.trans_features.shape[0]
        self.num_features = self.trans_features.shape[-1]
        self.num_responses = self.trans_responses.shape[-1]


def get_model_callbacks(config: configs.Config, existing_history: dict) -> List[keras.callbacks.Callback]:
    """Creates model callbacks from a bfgn config.

    Args:
        config: bfgn config.
        existing_history: Existing model training history if the model has already been partially or completely t rained.

    Returns:
        List of model callbacks.
    """
    callbacks = [
        HistoryCheckpoint(
            config=config,
            existing_history=existing_history,
            verbose=config.model_training.verbosity,
        ),
        #CustomMetric(),
    ]
    
#    if config.callback_cyclical_learning.use_callback:
#        callbacks.append(
#                CyclicalLearning(
#                config = config,
#                cycles = config.callback_cyclical_learning.cycles,
#                learning = config.callback_cyclical_learning.learning,
#                verbose = config.callback_cyclical_learning.verbose,
#                amplitude = config.callback_cyclical_learning.amplitude,
#            )
#        )
    if config.callback_general.save_best_model:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=experiments.get_model_filepath(config),
                monitor=config.callback_early_stopping.loss_metric,
                verbose=config.model_training.verbosity,
                save_best_only=True,
                mode='auto',
            )
        )
    elif config.callback_general.checkpoint_periods is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                experiments.get_model_filepath(config).replace(".h5","-{epoch:02d}-{val_loss:.2f}.h5"),
                period=config.callback_general.checkpoint_periods,
                verbose=config.model_training.verbosity,
            )
        )
    if config.callback_early_stopping.use_callback:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config.callback_early_stopping.loss_metric,
                min_delta=config.callback_early_stopping.min_delta,
                patience=config.callback_early_stopping.patience,
                restore_best_weights=True,
            )
        )
    if config.callback_reduced_learning_rate.use_callback:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=config.callback_reduced_learning_rate.loss_metric,
                factor=config.callback_reduced_learning_rate.factor,
                min_delta=config.callback_reduced_learning_rate.min_delta,
                patience=config.callback_reduced_learning_rate.patience,
            )
        )
    if config.callback_tensorboard.use_callback:
        dir_out = os.path.join(config.model_training.dir_out, _DIR_TENSORBOARD)
        callbacks.append(
            keras.callbacks.TensorBoard(
                dir_out,
                histogram_freq=config.callback_tensorboard.histogram_freq,
                write_graph=config.callback_tensorboard.write_graph,
                write_grads=config.callback_tensorboard.write_grads,
                write_images=config.callback_tensorboard.write_images,
                update_freq=config.callback_tensorboard.update_freq,
            )
        )
    if config.callback_general.use_terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    
    
    return callbacks
