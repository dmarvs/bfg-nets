from typing import Iterable
import keras.backend as K
import numpy as np
import os
import warnings

from rsCNN.networks import architectures, callbacks, history


class TrainingHistory(object):
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.
    """
    # TODO - Fabina, can you populate this with the useful info you want to retain from training?
    # TODO - Phil:  let's punt until we know what we need and what everything else looks like?
    pass


class NetworkConfig(object):
    """ A wrapper class designed to hold all relevant configuration information for the
        training of a new network.
    """


    # TODO: typedef loss_function 
    def __init__(self, network_type : str, loss_function, inshape : Iterable[int], n_classes : Iterable[int], **kwargs):
        """
          Arguments:
          network_type - str
            Style of the network to use.  Options are:
              flex_unet
              flat_regress_net
          loss_function - function
            Keras or tensor flow based loss function for the cnn.
          inshape - tuple/list
            Designates the input shape of an image to be passed to
            the network.
          n_classes - tuple/list
            Designates the output shape of targets to be fit by the network
        """
        self.network_type = network_type
        self.loss_function = loss_function
        self.inshape = inshape
        self.n_classes = n_classes


        if (self.network_type == 'flex_unet'):
            self.create_architecture = architectures.unet.flex_unet
            self.architecture_options = {
                'conv_depth': kwargs.get('conv_depth', 16),
                'batch_norm': kwargs.get('batch_norm', False),
            }
        elif (self.network_type == 'flat_regress_net'):
            self.create_architecture = architectures.regress_net.flat_regress_net
            self.architecture_options = {
                'conv_depth': kwargs.get('conv_depth', 16),
                'batch_norm': kwargs.get('batch_norm', False),
                'n_layers': kwargs.get('n_layers', 8),
                'conv_pattern': kwargs.get('conv_pattern', [3]),
                'output_activation': kwargs.get('output_activation', 'softmax'),
            }
        elif (self.network_type == 'residual_net'):
            self.create_architecture = architectures.residual_net.create_residual_network
            self.architecture_options = architectures.residual_net.parse_architecture_options(**kwargs)
        else:
            NotImplementedError('Invalid network type: ' + self.network_type)

        # Optional arguments
        # TODO:  Phil: I have dir_out which is supposed to control where models, history, etc are saved. Based on your
        # TODO:  comments this morning and the output_directory parameter for training arguments (below), I don't know
        # TODO:  if dir_out makes sense any longer. e.g., we expect filepath_model_out, filepath_history_out to have the
        # TODO:  full path now? We should refactor for whatever intentions we have.
        
        self.dir_out = kwargs.get('dir_out', './')
        self.filepath_model_out = kwargs.get('filepath_model_out', 'model.h5')
        self.filepath_history_out = kwargs.get('filepath_history_out', 'history.json')
        self.checkpoint_periods = kwargs.get('checkpoint_periods', 5)
        self.verbosity = kwargs.get('verbosity', 1)
        self.append_existing = kwargs.get('append_existing', False)

        # Training arguments
        self.batch_size = kwargs.get('batch_size', 1)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.n_noimprovement_repeats = kwargs.get('n_noimprovement_repeats', 10)
        self.output_directory = None  # TODO: give a default
        self.verification_fold = kwargs.get('verification_fold', None)

        # Callbacks
        self.callbacks_use_tensorboard = kwargs.get('callbacks_use_tensorboard', True)
        self.filepath_tensorboard_out = kwargs.get('dir_tensorboard_out', 'tensorboard')
        self.tensorboard_update_freq = kwargs.get('tensorboard_update_freq', 'epoch')
        self.tensorboard_histogram_freq = kwargs.get('tensorboard_histogram_freq', 0)
        self.tensorboard_write_graph = kwargs.get('tensorboard', True)
        self.tensorboard_write_grads = kwargs.get('tensorboard', False)
        self.tensorboard_write_images = kwargs.get('tensorboard', True)

        self.callbacks_use_early_stopping = kwargs.get('callbacks_use_early_stopping', True)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 10**-4)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 50)

        self.callbacks_use_reduced_learning_rate = kwargs.get('callbacks_use_reduced_learning_rate', True)
        self.reduced_learning_rate_factor = kwargs.get('reduced_learning_rate_factor', 0.5)
        self.reduced_learning_rate_min_delta = kwargs.get('reduced_learning_rate_min_delta', 10**-4)
        self.reduced_learning_rate_patience = kwargs.get('reduced_learning_rate_patience', 10)

        self.callbacks_use_terminate_on_nan = kwargs.get('terminate_on_nan', True)


class CNN():

    def __init__(self, network_config : NetworkConfig, load_history = True, reinitialize = False):
        """ Initializes the appropriate network

        Arguments:
        network_config - NetworkConfig
          Configuration parameter object for the network.

        Keyword Arguments:
        reinitialize - bool
          Flag directing whether the model should be re-initialized from scratch (no weights).
        load_history - bool
          Flag directing whether the model should load it's training history.
        """
        self.config = network_config
        # TODO:  how do we want to reload models? in init? in reload function? just putting outlined code here for now,
        # TODO:  and letting you make the decision about what's best given your experience with this workflow. Not sure
        # TODO:  if we want to explicitly check for existing model objects and assert that the user wants to load
        # TODO:  existing content, but this depends on the other decisions that are made
        # if (model objects are not saved at the config-specified locations) and (config.load_existing = False):

        if (load_history and not reinitialize):
            warning.warn('Warning: loading model history and re-initializing the model')
            
        #TODO: condense to one check 
        if (reinitialize == False):
            if (os.path.isfile(self.config.filepath_model_out)):
                self.model = keras.models.load_model(self.config.filepath_model_out)
            else:
                reinitialize = True
                
        # TODO: probably remove history load here and move it over to training
        if (reinitialize):
            self.model = self.config.create_architecture(self.config.inshape, self.config.n_classes, **self.config.architecture_options)
        if (load_history):
            if (os.path.isfile(self.config.filepath_history_out)): history.load_history(self.config.filepath_history_out)

        #TODO: finish adjusting history updates / optimizer settings

        # TODO: set optimizer as config param
        self.model.compile(loss=self.config.loss_function,optimizer='adam')

        self.history = dict()
        self.training = None
        # elif (model objects exist) and (config.load_existing = True):
        # self.model = keras.models.load_model(filepath depends on config decision above, custom_objects=TODO)
        # self._initial_epoch = len(self.history['lr'])  # Probably want this if training is continued
        # K.set_value(self.model.optimizer.lr, self.history['lr'][-1])  # Probably don't want this

    def calculate_training_memory_usage(self, batch_size):
        # Shamelessly copied from
        # https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
        # but not tested rigorously
        shapes_mem_count = 0
        for l in self.model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

    # TODO during fit, make sure that all training_options (as well as network options) are saved with the model
    def fit(self, features, responses, fold_assignments):
        model_callbacks = callbacks.get_callbacks(self.config)

        if (self.config.verification_fold is not None):
            train_subset = fold_assignments == self.config.verification_fold
            test_subset = np.logical_not(train_subset)
            train_features = features[train_subset, ...]
            train_responses = responses[train_subset]
            validation_data = (features[test_subset, ...], responses[test_subset, ...])
        else:
            train_features = features
            train_responses = responses
            validation_data = None

        self.model.fit(train_features,
                       train_responses,
                       validation_data=validation_data,
                       epochs=self.config.max_epochs,
                       batch_size=self.config.batch_size,
                       verbose=self.config.verbosity,
                       shuffle=False,
                       callbacks=model_callbacks)

    def fit_sequence(self, train_sequence, validation_sequence=None):
        # TODO:  reimplement if/when we need generators, ignore for now
        raise NotImplementedError

    def predict(self, features):
        return self.model.predict(features, batch_size=self.config.batch_size, verbose=self.config.verbosity)

    def predict_sequence(self, predict_sequence):
        # TODO:  reimplement if/when we need generators, ignore for now
        raise NotImplementedError
