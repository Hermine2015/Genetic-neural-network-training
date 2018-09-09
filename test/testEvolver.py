from unittest import TestCase
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration
from app.evolver import Evolver

class TestEvolver(TestCase):
    def test_give_an_individual_when_get_configuration_from_individual_then_a_config_should_be_returned(self):
        individual = [ 5, 8, 2, 3, 2, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0 ]

        configuration = ConvolutionalConfiguration(
            total_convolutional_layers=5,
            total_convolutional_filters=8,
            filter_size_convolution=(3, 3),
            filter_size_deconvolution=(2, 2),
            pool_size=(2, 2),
            strides=(2, 2),
            activation='relu',
            padding='same',
            output_activation='sigmoid',
            optimizer='adam',
            loss='binary_crossentropy'
        )

        result = Evolver()._get_configuration_from_individual(individual)

        self.assertEqual(configuration.total_convolutional_layers, result.total_convolutional_layers)
        self.assertEqual(configuration.total_convolutional_filters, result.total_convolutional_filters)
        self.assertEqual(configuration.filter_size_convolution, result.filter_size_convolution)
        self.assertEqual(configuration.filter_size_deconvolution, result.filter_size_deconvolution)
        self.assertEqual(configuration.pool_size, result.pool_size)
        self.assertEqual(configuration.strides, result.strides)
        self.assertEqual(configuration.activation, result.activation)
        self.assertEqual(configuration.padding, result.padding)
        self.assertEqual(configuration.output_activation, result.output_activation)
        self.assertEqual(configuration.optimizer, result.optimizer)
        self.assertEqual(configuration.loss, result.loss)