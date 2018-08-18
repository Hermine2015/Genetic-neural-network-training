from deap import creator, tools, base


class ToolboxGenerator:

    def register_all_from_configurations(self, toolbox, configurations):
        [ self.register_from_configuration(toolbox, configuration) for configuration in configurations]

    def register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    def initialise_individuals(self, toolbox, configurations,  fitness):
        creator.create("Individual", list, fitness=fitness)

        attributes_to_evolve = self.get_attributes_to_evolve_from_configurations(toolbox, configurations)

        toolbox.register("individual", tools.initCycle, creator.Individual, attributes_to_evolve, n=len(attributes_to_evolve))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def add_fitness_to_creator(self, creator, desired_scores=(-1.0)):
        creator.create("FitnessMulti", base.Fitness, weights=desired_scores)

    def get_attributes_to_evolve_from_configurations(self, toolbox, configurations):
        attributes_as_list = [
            getattr(toolbox, configuration.name)
            for configuration in configurations
        ]

        return tuple(attributes_as_list)