import random

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