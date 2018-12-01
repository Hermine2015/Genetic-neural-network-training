from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration
from keras.callbacks import EarlyStopping
from app.scoring.scorer import MeanIntersectionOverUnion
from app.evolver import Evolver

class NeuralNetworkEvolver(Evolver):
    def evaluate(self, individual):

        evaluator = [ MeanIntersectionOverUnion().get_score ]

        configuration = self._get_configuration_from_individual(individual)

        model = ConvolutionalNeuralNetwork().get_model(self.dimensions, evaluator, configuration)

        #TODO extract these params to some config
        earlystopper = EarlyStopping(patience=5, verbose=0)

        history = model.fit(self.training_set, self.training_label,
                            validation_split=0.1, batch_size=8, epochs=30,
                            callbacks=[earlystopper], verbose=0)

        print(
            '\n-----------EVALUATING-------------\n' +
            self._get_configuration_as_string_from_individual(individual) +
            '\n#### SCORE: {} ####'.format(history.history['get_score'][-1]) +
            '\n-------------------------------'
        )

        return tuple([history.history['get_score'][-1]])

    def _get_configuration_as_string_from_individual(self, individual):
        configuration = self._get_configuration_from_individual(individual)

        labels = '\n'.join(['total_convolutional_layers: {}', 'total_convolutional_filters: {}',
                            'filter_size_convolution: {}', 'filter_size_deconvolution: {}', 'pool_size: {}',
                            'strides: {}', 'activation: {}', 'padding: {}', 'optimizer: {}',
                            'output_activation: {}', 'loss: {}'])

        return labels.format(configuration.total_convolutional_layers, configuration.total_convolutional_filters,
                             configuration.filter_size_convolution, configuration.filter_size_deconvolution,
                             configuration.pool_size, configuration.strides, configuration.activation,
                             configuration.padding, configuration.optimizer, configuration.output_activation,
                             configuration.loss)
        
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
