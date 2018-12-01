from app.evolution.toolbox_generator import ToolboxGenerator
import numpy as np
from deap import algorithms
from app.data_preparation.image_preparer import ObjectRecognitionImagePreparer
import multiprocessing

class Evolver:
    def __init__(self, training_path, label_path, dimensions, data_limit=None):
        self.training_set, self.training_label = \
            ObjectRecognitionImagePreparer().get_resized_training_data(training_path, label_path, dimensions)

        if data_limit is not None:
            self.training_set = self.training_set[:data_limit]
            self.training_label = self.training_label[:data_limit]

        self.dimensions = dimensions

    def evolve(self, evolution_configuration):
        toolbox = ToolboxGenerator().get_toolbox(evolution_configuration, self.evaluate)

        pool = multiprocessing.Pool(processes=evolution_configuration.total_threads)

        toolbox.register("map", pool.map)

        pop = toolbox.population(n=evolution_configuration.population)

        total_generations = evolution_configuration.generations

        algorithms.eaSimple(
            pop, toolbox,
            evolution_configuration.crossover['probability'],
            evolution_configuration.mutation['probability'],
            total_generations)

        top = sorted(pop, key=lambda x: x.fitness.values[0])[-1]

        return self._get_configuration_as_string_from_individual(top)

    def evaluate(self, individual):
        pass

    def _make_even(self, value):
        rounded_value = self._get_int(value)

        return rounded_value if rounded_value % 2 == 0 else rounded_value + 1

    def _get_configuration_as_string_from_individual(self, individual):
        pass
        
    def _get_configuration_from_individual(self, individual):
        pass

    def _get_int(self, value):
        return int(round(value))

    def _get_tuple(self, dimensions, values):
        return tuple(np.repeat(self._get_int(values), self._get_int(dimensions)))
