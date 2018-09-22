from app.evolution.evolution_configuration import EvolutionConfiguration, ToolboxConfiguration
import random
from app.evolver import Evolver

evolution_configuration = EvolutionConfiguration(
    [
        ToolboxConfiguration('total_convolutional_layers', random.randint, 2, 8), # for more layers, need to increase dimensions of images
        ToolboxConfiguration('total_convolutional_filters', random.randint, 2, 20),
        ToolboxConfiguration('filter_size_convolution_value', random.randint, 4, 8),
        ToolboxConfiguration('filter_size_deconvolution_value', random.randint, 2, 4),
        ToolboxConfiguration('pool_size_value', random.randint, 1, 2),
        ToolboxConfiguration('strides_value', random.randint, 1, 2),
        ToolboxConfiguration('activation', random.randint, 0, 1),
        ToolboxConfiguration('padding', random.randint, 0, 1),
        ToolboxConfiguration('output_activation', random.randint, 0, 1),
        ToolboxConfiguration('optimizer', random.randint, 0, 1),
        ToolboxConfiguration('loss', random.randint, 0, 1)
    ],
    [
        { "name": "accuracy", "minimize": False }
    ],
    { "name": "Gaussian", "mu": 0.0, "sigma": 0.2, "indpb": 0.2, "probability": 0.3 },
    { "name": "Two-point", "probability": 0.5 },
    { "name": "Tournament", "tournament-size": 3 }
)

if __name__ == "__main__":
    training_path = '/Users/georgieva_kristina/repositories/personal/data/train/images'
    label_path = '/Users/georgieva_kristina/repositories/personal/data/train/masks'
    dimensions = (128, 128, 1)

    evolver = Evolver(training_path, label_path, dimensions)

    evolver.evolve(evolution_configuration)