import random as rd

from .population import Population
from .individual import Individual

class Generator:

    def random_binary_population(chromosome_size, population_size):
        return Population([Individual([rd.randint(0, 1) for j in range(chromosome_size)], None) for i in range(population_size)])