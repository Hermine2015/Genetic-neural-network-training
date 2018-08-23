from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import concatenate

class ConvolutionalNeuralNetwork:

    def get_model(self,
                  input_dimentions,
                  metrics,
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
                  ):

        total_deconvolutional_filters = self._get_deconvolutional_filters(total_convolutional_layers,
                                                                    total_convolutional_filters)
        total_deconvolutional_layers = total_convolutional_layers - 1

        inputs = Input(input_dimentions)
        input_layer = Lambda(lambda x: x / 255)(inputs)

        convolutional_layers = self._setup_convolution(input_layer, total_convolutional_layers, total_convolutional_filters,
                                                 filter_size_convolution, pool_size, activation, padding)
        last_layer = self._setup_deconvolution(convolutional_layers, total_deconvolutional_layers,
                                         total_deconvolutional_filters,
                                         filter_size_deconvolution, activation, padding, strides,
                                         filter_size_deconvolution)

        outputs = Convolution2D(1, (1, 1), activation=output_activation)(last_layer)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


    def _setup_convolution(self, input_layer, total_layers, total_filters, filter_size,
                           pool_size, activation, padding):

        previous_layer = input_layer

        convolutional_layers = []

        for layer_index in range(0, total_layers):
            print('Creating convolution layer ' + str(layer_index))

            convolutional_layer = Convolution2D(total_filters, filter_size, activation=activation,
                                                padding=padding)(previous_layer)
            convolutional_layer = Convolution2D(total_filters, filter_size, activation=activation,
                                                padding=padding)(convolutional_layer)

            if layer_index < total_layers - 1:
                pooling_layer = MaxPooling2D(pool_size)(convolutional_layer)
                previous_layer = pooling_layer
            else:
                previous_layer = convolutional_layer

            convolutional_layers.append(convolutional_layer)
            total_filters = total_filters * 2

        return convolutional_layers

    def _setup_deconvolution(self, covolutional_layers, total_layers, total_filters, filter_size_convolution,
                             activation, padding, strides, filter_size_deconvolution):

        previous_layer = covolutional_layers[len(covolutional_layers) - 1]

        deconvolutional_layers = []
        merged_layers = []

        for layer_index in range(0, total_layers):
            print('Creating deconvolution layer ' + str(layer_index))
            merged_layer = Conv2DTranspose(total_filters, filter_size_deconvolution, strides=strides, padding=padding)(
                previous_layer)
            convolution_to_reverse = covolutional_layers[len(covolutional_layers) - layer_index - 2]
            merged_layer = concatenate([merged_layer, convolution_to_reverse])

            convolution_layer = Convolution2D(total_filters, filter_size_convolution, activation=activation,
                                              padding=padding)(merged_layer)
            convolution_layer = Convolution2D(total_filters, filter_size_convolution, activation=activation,
                                              padding=padding)(convolution_layer)

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

