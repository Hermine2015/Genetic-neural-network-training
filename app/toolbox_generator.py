from deap import creator, tools, base


class ToolboxGenerator:
    def get_toolbox(self, evolution_configuration, evaluation_function):
        gene_configurations = evolution_configuration.genes

        toolbox = base.Toolbox()

        self._register_all_from_configurations(toolbox, gene_configurations)
        self._add_fitness_to_creator(creator, name="FitnessMulti") #todo change to get min and may from config

        fitness = self._get_from_toolbox(creator, "FitnessMulti")
        self._initialise_individuals(toolbox, gene_configurations, fitness)

        self._register_mutation(toolbox, evolution_configuration)
        self._register_crossover(toolbox, evolution_configuration)
        self._register_selection(toolbox, evolution_configuration)
        self._register_evaluation(toolbox, evaluation_function)

        return toolbox

    def _register_all_from_configurations(self, toolbox, configurations):
        [self._register_from_configuration(toolbox, configuration) for configuration in configurations]

    def _register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    def _initialise_individuals(self, toolbox, configurations, fitness):
        creator.create("Individual", list, fitness=fitness)

        attributes_to_evolve = self._get_attributes_to_evolve_from_configurations(toolbox, configurations)

        toolbox.register("individual", tools.initCycle, creator.Individual, attributes_to_evolve, n=len(attributes_to_evolve))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _add_fitness_to_creator(self, creator, desired_scores=(-1.0), name="FitnessMulti"):
        creator.create(name, base.Fitness, weights=desired_scores)

    def _get_attributes_to_evolve_from_configurations(self, toolbox, configurations):
        attributes_as_list = [
            self._get_from_toolbox(toolbox, configuration.name)
            for configuration in configurations
        ]

        return tuple(attributes_as_list)

    def _register_mutation(self, toolbox, evolution_configuration):
        mutation_dictionary = evolution_configuration.mutation
        name = mutation_dictionary["name"]

        if name == "Gaussian":
            self._register_gaussian_mutation(toolbox, mutation_dictionary)
        elif name == "Flip-bit":
            self._register_flipbit_mutation(toolbox, mutation_dictionary)
        else:
            raise Exception("Unsuported mutation type: {}".format(name))

    def _register_crossover(self, toolbox, evolution_configuration):
        crossover_dictionary = evolution_configuration.crossover
        name = crossover_dictionary["name"]

        if name == "Two-point":
            self._register_two_point_crossover(toolbox)
        else:
            raise Exception("Unsuported crossover type: {}".format(name))

    def _register_selection(self, toolbox, evolution_configuration):
        selection_dictionary = evolution_configuration.selection
        name = selection_dictionary["name"]

        if name == "Tournament":
            self._register_tournament_selection(toolbox, selection_dictionary)
        else:
            raise Exception("Unsuported selection type: {}".format(name))

    def _register_gaussian_mutation(self, toolbox, parameters):
        toolbox.register("mutate", tools.mutGaussian,
                         mu=parameters["mu"], sigma=parameters["sigma"], indpb=parameters["indpb"])

    def _register_flipbit_mutation(self, toolbox, parameters):
        toolbox.register("mutate", tools.mutFlipBit, indpb=parameters["indpb"])

    def _register_two_point_crossover(self, toolbox):
        toolbox.register("mate", tools.cxTwoPoint)

    def _register_tournament_selection(self, toolbox, parameters):
        toolbox.register("select", tools.selTournament, tournsize=parameters["tournament-size"])

    def _register_evaluation(self, toolbox, evaluation_function):
        toolbox.register("evaluate", evaluation_function)

    def _get_from_toolbox(self, toolbox, attribute_name):
        return getattr(toolbox, attribute_name)
