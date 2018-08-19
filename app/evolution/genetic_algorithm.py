from deap import algorithms

class GeneticAlgorithm:

    def evolve(self,
               toolbox,
               mutation_probability=0.3,
               crossover_probability=0.5,
               total_generations=25,
               population_size=50):

        population = toolbox.population(n=population_size)

        algorithms.eaSimple(
            population,
            toolbox,
            mutation_probability,
            crossover_probability,
            total_generations)

        best_individual = sorted(population, key=lambda x: x.fitness.values[0])[-1]

        return best_individual