from deap import creator, tools, base

class ToolboxGenerator:
    def register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    def initialise_individuals(self, toolbox, attributes_to_evolve, individual_size, fitness):
        creator.create("Individual", list, fitness=fitness)

        toolbox.register("individual", tools.initCycle, creator.Individual, attributes_to_evolve, n=individual_size)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def add_fitness_to_creator(self, creator, desired_scores=(-1.0)):
        creator.create("FitnessMulti", base.Fitness, weights=desired_scores)