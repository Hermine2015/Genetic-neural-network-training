from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from app.evolution.toolbox_generator import ToolboxGenerator
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration
import numpy as np
from deap import algorithms
from app.data_preparation.image_preparer import ObjectRecognitionImagePreparer
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from app.scoring.scorer import MeanIntersectionOverUnion
import multiprocessing

class Evolver:
    def __init__(self, training_path, label_path, dimensions):
        self.training_set, self.training_label = \
            ObjectRecognitionImagePreparer().get_resized_training_data(training_path, label_path, dimensions)

        # TODO add limit to total training data via config

        self.dimensions = dimensions

    def evolve(self, evolution_configuration):
        toolbox = ToolboxGenerator().get_toolbox(evolution_configuration, self.evaluate)

        # TODO might need to move to __main__
        pool = multiprocessing.Pool(processes=10) #TODO increase

        toolbox.register("map", pool.map)

        # TODO get params from config
        pop = toolbox.population(n=50)

        total_generations = 25

        algorithms.eaSimple(
            pop, toolbox,
            .3, .5,
            total_generations)

        top = sorted(pop, key=lambda x: x.fitness.values[0])[-1]

        return top

    def evaluate(self, individual):
        evaluator = [ MeanIntersectionOverUnion().get_score ]

        configuration = self._get_configuration_from_individual(individual)

        model = ConvolutionalNeuralNetwork().get_model(self.dimensions, evaluator, configuration)

        #TODO extract these params to config
        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)

        history = model.fit(self.training_set, self.training_label,
                            validation_split=0.1, batch_size=8, epochs=30,
                            callbacks=[earlystopper, checkpointer])

        return tuple([history.history['get_score'][-1]])

    def _make_even(self, value):
        rounded_value = self._get_int(value)

        return rounded_value if rounded_value % 2 == 0 else rounded_value + 1

    def _get_configuration_from_individual(self, individual):
        return ConvolutionalConfiguration(
            total_convolutional_layers=self._get_int(individual[0]),
            total_convolutional_filters=self._get_int(individual[1]),
            filter_size_convolution=self._get_tuple(2, self._make_even(individual[2])),
            filter_size_deconvolution= self._get_tuple(2, individual[3]),
            pool_size=self._get_tuple(2, self._make_even(individual[4])),
            strides=self._get_tuple(2, self._make_even(individual[5])),
            activation=self._get_activation(individual[6]),
            padding=self._get_padding(individual[7]),
            output_activation=self._get_activation(individual[8]),
            optimizer=self._get_optimizer(individual[9]),
            loss=self._get_loss(individual[10])
        )

    def _get_int(self, value):
        return int(round(value))

    def _get_tuple(self, dimensions, values):
        return tuple(np.repeat(self._get_int(values), self._get_int(dimensions)))

    def _get_activation(self, value):
        if self._get_int(value) == 0:
            return 'relu'
        elif self._get_int(value) == 1:
            return 'sigmoid'

    def _get_padding(self, value):
        return 'same'

    def _get_optimizer(self, value):
        return 'adam'

    def _get_loss(self, value):
        return 'binary_crossentropy'
