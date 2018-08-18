class EvolutionConfiguration:
    def __init__(self,
                 genes,
                 scores,
                 mutation,
                 crossover,
                 selection
                 ):
        self.genes = genes
        self.scores = scores
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection