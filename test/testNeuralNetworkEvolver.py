from unittest import TestCase
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration
from app.neural_network_evolver import NeuralNetworkEvolver
from unittest.mock import patch
from unittest.mock import MagicMock
from keras.callbacks import EarlyStopping

class TestEvolver(TestCase):

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_give_an_individual_when_get_configuration_from_individual_then_a_config_should_be_returned(self, image_preparer_mock):
        image_preparer_mock.return_value = [], []

        individual, configuration = self.get_configuration_mock()

        result = NeuralNetworkEvolver(None, None, None)._get_configuration_from_individual(individual)

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

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_activation_integer_close_to_zero_when_activation_then_relu_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = NeuralNetworkEvolver(None, None, None)._get_activation(0.2)

        self.assertEqual('relu', result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_activation_integer_close_to_one_when_activation_then_sigmoid_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = NeuralNetworkEvolver(None, None, None)._get_activation(0.9)

        self.assertEqual('sigmoid', result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_a_padding_integer_close_to_zero_when_get_padding_same_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = NeuralNetworkEvolver(None, None, None)._get_padding(0.2)

        self.assertEqual('same', result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_optimizer_integer_close_to_zero_when_get_optimizer_then_adam_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = NeuralNetworkEvolver(None, None, None)._get_optimizer(0.2)

        self.assertEqual('adam', result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_a_loss_integer_close_to_zero_when_get_loss_then_binary_crossentropy_should_be_returned(
            self, preparation_mock):
        preparation_mock.return_value = [], []

        result = NeuralNetworkEvolver(None, None, None)._get_loss(0.2)

        self.assertEqual('binary_crossentropy', result)

    @patch('app.neural_network_evolver.NeuralNetworkEvolver._get_configuration_from_individual')
    @patch('keras.callbacks.EarlyStopping.__init__')
    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    @patch('app.scoring.scorer.MeanIntersectionOverUnion.get_score')
    @patch('app.neural_network.convolutional_neural_network.ConvolutionalNeuralNetwork.get_model')
    def test_given_an_individual_when_evaluate_then_the_neural_network_should_be_ran(
            self, network_mock, score_mock, preparation_mock, early_stopping_mock, config_mock):

        individual, configuration = self.get_configuration_mock()

        preparation_mock.return_value = ['training'], ['label']
        model_mock = MagicMock()
        network_mock.return_value = model_mock
        config_mock.return_value = configuration
        score_mock.return_value = 0.76
        dimensions = (124, 124, 1)
        model_mock.fit.return_value = History()

        early_stopping_mock.return_value = None

        result = NeuralNetworkEvolver(None, None, dimensions).evaluate(individual)

        network_mock.assert_called_with(dimensions, [score_mock], configuration)
        model_mock.fit.assert_called_with(['training'], ['label'],
                            validation_split=0.1, batch_size=8, epochs=30,
                            callbacks=[Any(EarlyStopping)], verbose=0)

        self.assertEqual((0.76,), result)

    @patch('app.data_preparation.image_preparer.ObjectRecognitionImagePreparer.get_resized_training_data')
    def test_given_an_individual_when_get_configuration_as_string_from_individual_a_formatted_String_should_be_returned(
            self, preparation_mock):

        preparation_mock.return_value = ['training'], ['label']

        individual = [5, 8, 3, 2, 2, 2, 0, 0, 1, 0, 0]

        config_as_string = NeuralNetworkEvolver(None, None, (124, 124, 1))._get_configuration_as_string_from_individual(individual)

        expected = 'total_convolutional_layers: 5\n' \
                   'total_convolutional_filters: 8\n' \
                   'filter_size_convolution: (4, 4)\n' \
                   'filter_size_deconvolution: (2, 2)\n' \
                   'pool_size: (2, 2)\n' \
                   'strides: (2, 2)\n' \
                   'activation: relu\n' \
                   'padding: same\n' \
                   'optimizer: adam\n' \
                   'output_activation: sigmoid\n' \
                   'loss: binary_crossentropy'

        self.assertEqual(config_as_string, expected)

    def get_configuration_mock(self):
        individual = [5, 8, 3, 2, 2, 2, 0, 0, 1, 0, 0]

        configuration = ConvolutionalConfiguration(
            total_convolutional_layers=5,
            total_convolutional_filters=8,
            filter_size_convolution=(4, 4),
            filter_size_deconvolution=(2, 2),
            pool_size=(2, 2),
            strides=(2, 2),
            activation='relu',
            padding='same',
            output_activation='sigmoid',
            optimizer='adam',
            loss='binary_crossentropy'
        )

        return individual, configuration

class History:
    def __init__(self):
        self.history = {
            'get_score': [0, 0.2, 0.21, 0.19, 0.5, 0.76]
        }

def Any(cls):
    class Any(cls):
        def __eq__(self, other):
            return True

    return Any()