import random

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

class ToolboxConfiguration:
    def __init__(self,
                 name,
                 type=random.randint,
                 lower_bound=0,
                 upper_bound=1
                 ):
        self.name = name
        self.type = type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound