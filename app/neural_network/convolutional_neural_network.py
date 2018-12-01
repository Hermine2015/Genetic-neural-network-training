from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import concatenate

class ConvolutionalNeuralNetwork:

    def get_model(self,
                  input_dimentions,
                  metrics,
                  configuration
                  ):

        total_deconvolutional_filters = self._get_deconvolutional_filters(configuration.total_convolutional_layers,
                                                                          configuration.total_convolutional_filters)
        total_deconvolutional_layers = configuration.total_convolutional_layers - 1

        inputs = Input(input_dimentions)
        input_layer = Lambda(lambda x: x / 255)(inputs)

        convolutional_layers = self._setup_convolution(input_layer, configuration)
        last_layer = self._setup_deconvolution(convolutional_layers, total_deconvolutional_layers,
                                               total_deconvolutional_filters, configuration)

        outputs = Convolution2D(1, (1, 1), activation=configuration.output_activation)(last_layer)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=configuration.optimizer, loss=configuration.loss, metrics=metrics)

        return model


    def _setup_convolution(self, input_layer, configuration):

        previous_layer = input_layer

        convolutional_layers = []

        total_filters = configuration.total_convolutional_filters

        for layer_index in range(0, configuration.total_convolutional_layers):
            convolutional_layer = Convolution2D(total_filters, configuration.filter_size_convolution,
                                                activation=configuration.activation, padding=configuration.padding)(previous_layer)
            convolutional_layer = Convolution2D(total_filters, configuration.filter_size_convolution,
                                                activation=configuration.activation, padding=configuration.padding)(convolutional_layer)

            if layer_index < configuration.total_convolutional_layers - 1:
                pooling_layer = MaxPooling2D(configuration.pool_size)(convolutional_layer)
                previous_layer = pooling_layer
            else:
                previous_layer = convolutional_layer

            convolutional_layers.append(convolutional_layer)
            total_filters = total_filters * 2

        return convolutional_layers

    def _setup_deconvolution(self, covolutional_layers, total_layers, total_filters, configuration):

        previous_layer = covolutional_layers[len(covolutional_layers) - 1]

        deconvolutional_layers = []
        merged_layers = []

        for layer_index in range(0, total_layers):
            merged_layer = Conv2DTranspose(total_filters, configuration.filter_size_deconvolution,
                                           strides=configuration.strides, padding=configuration.padding)(previous_layer)
            convolution_to_reverse = covolutional_layers[len(covolutional_layers) - layer_index - 2]
            merged_layer = concatenate([merged_layer, convolution_to_reverse])

            convolution_layer = Convolution2D(total_filters, configuration.filter_size_convolution, activation=configuration.activation,
                                              padding=configuration.padding)(merged_layer)
            convolution_layer = Convolution2D(total_filters, configuration.filter_size_convolution, activation=configuration.activation,
                                              padding=configuration.padding)(convolution_layer)

            deconvolutional_layers.append(convolution_layer)
            merged_layers.append(merged_layer)

            previous_layer = merged_layer
            total_filters = int(total_filters / 2)

        last_layer = previous_layer

        return last_layer


    def _get_deconvolutional_filters(self, total_convolutional_layers, total_convolutional_filters):
        return [i * 2 ** exp
                for exp in range(0, total_convolutional_layers - 1)
                for i in range(total_convolutional_filters, total_convolutional_filters + 1)][-1]

