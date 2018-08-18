from deap import creator, tools, base

class ToolboxGenerator:
    def get_toolbox(self, evolution_configuration):
        configurations = evolution_configuration["genes"]

        toolbox = base.Toolbox()

        self.register_all_from_configurations(toolbox, configurations)
        self.add_fitness_to_creator(creator, name="FitnessMulti") #todo change to get min and may from config

        fitness = self._get_from_toolbox(creator, "FitnessMulti")
        self.initialise_individuals(toolbox, configurations, fitness)

        return toolbox


    def register_all_from_configurations(self, toolbox, configurations):
        [self._register_from_configuration(toolbox, configuration) for configuration in configurations]

    def _register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    def initialise_individuals(self, toolbox, configurations, fitness):
        creator.create("Individual", list, fitness=fitness)

        attributes_to_evolve = self._get_attributes_to_evolve_from_configurations(toolbox, configurations)

        toolbox.register("individual", tools.initCycle, creator.Individual, attributes_to_evolve, n=len(attributes_to_evolve))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def add_fitness_to_creator(self, creator, desired_scores=(-1.0), name="FitnessMulti"):
        creator.create(name, base.Fitness, weights=desired_scores)

    def _get_attributes_to_evolve_from_configurations(self, toolbox, configurations):
        attributes_as_list = [
            self._get_from_toolbox(toolbox, configuration.name)
            for configuration in configurations
        ]

        return tuple(attributes_as_list)

    def _get_from_toolbox(self, toolbox, attribute_name):
        return getattr(toolbox, attribute_name)
