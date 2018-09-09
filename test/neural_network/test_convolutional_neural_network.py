from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from keras import backend as K

from tensorflow.test import TestCase
from app.neural_network.convolutional_configuration import ConvolutionalConfiguration

class TestNeuralNetwork(TestCase):
    def test_given_no_parameters_when_get_model_then_a_model_with_the_default_convolution_parameters_should_be_setup(self):
        input_dimensions = (128, 128, 1)

        def mean_iou(y_true, y_pred):
            return K.mean(K.stack([]), axis=0)

        with self.test_session() as sess:
            model = ConvolutionalNeuralNetwork().get_model(input_dimensions, [mean_iou], self._get_default_configuration())

            layers = model._nodes_by_depth.values()

            outputs = {}

            for x in layers:
                nodes = x[0].inbound_layers

                if len(nodes) > 0:
                    for node in nodes:
                        outputs[node.name] = node.output_shape[3]

            self.assertEqual(8, outputs['conv2d_1'])
            self.assertEqual(8, outputs['conv2d_2'])
            self.assertEqual(16, outputs['conv2d_3'])
            self.assertEqual(16, outputs['conv2d_4'])
            self.assertEqual(32, outputs['conv2d_5'])
            self.assertEqual(32, outputs['conv2d_6'])
            self.assertEqual(64, outputs['conv2d_7'])
            self.assertEqual(64, outputs['conv2d_8'])
            self.assertEqual(128, outputs['conv2d_9'])
            self.assertEqual(128, outputs['conv2d_10'])

            self.assertEqual('relu', model.layers[2].activation.__name__)
            self.assertEqual('relu', model.layers[3].activation.__name__)
            self.assertEqual('relu', model.layers[5].activation.__name__)
            self.assertEqual('relu', model.layers[6].activation.__name__)
            self.assertEqual('relu', model.layers[8].activation.__name__)
            self.assertEqual('relu', model.layers[9].activation.__name__)
            self.assertEqual('relu', model.layers[11].activation.__name__)
            self.assertEqual('relu', model.layers[12].activation.__name__)
            self.assertEqual('relu', model.layers[14].activation.__name__)
            self.assertEqual('relu', model.layers[15].activation.__name__)

            self.assertEqual([3, 3, 1, 8], sess.graph.get_tensor_by_name("conv2d_1/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 8, 8], sess.graph.get_tensor_by_name("conv2d_2/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 8, 16], sess.graph.get_tensor_by_name("conv2d_3/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 16, 16], sess.graph.get_tensor_by_name("conv2d_4/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 16, 32], sess.graph.get_tensor_by_name("conv2d_5/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 32, 32], sess.graph.get_tensor_by_name("conv2d_6/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 32, 64], sess.graph.get_tensor_by_name("conv2d_7/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 64, 64], sess.graph.get_tensor_by_name("conv2d_8/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 64, 128], sess.graph.get_tensor_by_name("conv2d_9/kernel:0").shape.as_list())
            self.assertEqual([3, 3, 128, 128], sess.graph.get_tensor_by_name("conv2d_10/kernel:0").shape.as_list())


    def test_given_no_parameters_when_get_model_then_a_model_with_the_default_deconvolution_parameters_should_be_setup(self):
        input_dimensions = (128, 128, 1)

        def mean_iou(y_true, y_pred):
            return K.mean(K.stack([]), axis=0)

        with self.test_session() as sess:
            model = ConvolutionalNeuralNetwork().get_model(input_dimensions, [mean_iou], self._get_default_configuration())

            layers = model._nodes_by_depth.values()

            outputs = {}

            for x in layers:
                nodes = x[0].inbound_layers

                if len(nodes) > 0:
                    for node in nodes:
                        outputs[node.name] = node.output_shape[3]

            self.assertEqual(64, outputs['conv2d_transpose_1'])
            self.assertEqual(32, outputs['conv2d_transpose_2'])
            self.assertEqual(16, outputs['conv2d_transpose_3'])
            self.assertEqual(8, outputs['conv2d_transpose_4'])

            self.assertEqual([2, 2, 64, 128],
                             sess.graph.get_tensor_by_name("conv2d_transpose_1/kernel:0").shape.as_list())
            self.assertEqual([2, 2, 32, 128],
                             sess.graph.get_tensor_by_name("conv2d_transpose_2/kernel:0").shape.as_list())
            self.assertEqual([2, 2, 16, 64],
                             sess.graph.get_tensor_by_name("conv2d_transpose_3/kernel:0").shape.as_list())
            self.assertEqual([2, 2, 8, 32],
                             sess.graph.get_tensor_by_name("conv2d_transpose_4/kernel:0").shape.as_list())


    def test_given_no_parameters_when_get_model_then_a_model_with_the_default_output_parameters_should_be_setup(
            self):
        input_dimensions = (128, 128, 1)

        def mean_iou(y_true, y_pred):
            return K.mean(K.stack([]), axis=0)

        with self.test_session() as sess:
            model = ConvolutionalNeuralNetwork().get_model(input_dimensions, [mean_iou], self._get_default_configuration())

            layers = model._nodes_by_depth.values()

            outputs = {}

            for x in layers:
                nodes = x[0].inbound_layers

                if len(nodes) > 0:
                    for node in nodes:
                        outputs[node.name] = node.output_shape[3]


            self.assertEqual('sigmoid', model.layers[len(model.layers) - 1].activation.__name__)
            self.assertEqual('Adam', type(model.optimizer).__name__)
            self.assertEqual('binary_crossentropy', model.loss)

    def _get_default_configuration(self):
        return ConvolutionalConfiguration(
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
