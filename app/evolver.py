from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from app.scoring.scorer import MeanIntersectionOverUnion
from app.evolution.toolbox_generator import ToolboxGenerator
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration
import numpy as np

class Evolver:
    def evolve(self):
        toolbox_generator = ToolboxGenerator().get_toolbox()

    def evaluate(self):
        dimensions = (128, 128, 1)
        evaluator = MeanIntersectionOverUnion().get_score

        model = ConvolutionalNeuralNetwork().get_model(dimensions, evaluator)

    def _get_configuration_from_individual(self, individual):
        return ConvolutionalConfiguration(
            total_convolutional_layers=individual[0],
            total_convolutional_filters=individual[1],
            filter_size_convolution=self._get_tuple(individual[2], individual[3]),
            filter_size_deconvolution=self._get_tuple(individual[4], individual[5]),
            pool_size=self._get_tuple(individual[6], individual[7]),
            strides=self._get_tuple(individual[8], individual[9]),
            activation=self._get_activation(individual[10]),
            padding=self._get_padding(individual[11]),
            output_activation=self._get_activation(individual[12]),
            optimizer=self._get_optimizer(individual[13]),
            loss=self._get_loss(individual[14])
        )

    def _get_tuple(self, dimensions, values):
        return tuple(np.repeat(values, dimensions))

    def _get_activation(self, value):
        if value == 0:
            return 'relu'
        elif value == 1:
            return 'sigmoid'

    def _get_padding(self, value):
        return 'same'

    def _get_optimizer(self, value):
        return 'adam'

    def _get_loss(self, value):
        return 'binary_crossentropy'
