from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork

class Evolver:
    def evolve(self):
        dimensions = (128, 128, 1)

        model = ConvolutionalNeuralNetwork().get_model(dimensions)