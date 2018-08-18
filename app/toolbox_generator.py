from deap import creator, tools
class ToolboxGenerator:
    def register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    def initialise_individuals(self, toolbox, attributes_to_evolve, individual_size, fitness):
        creator.create("Individual", list, fitness=fitness)

        toolbox.register("individual", tools.initCycle, creator.Individual, attributes_to_evolve, n=individual_size)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)