from app.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from app.scoring.scorer import MeanIntersectionOverUnion
class Evolver:
    def evolve(self):
        dimensions = (128, 128, 1)
        evaluator = MeanIntersectionOverUnion().get_score

        model = ConvolutionalNeuralNetwork().get_model(dimensions, evaluator)
