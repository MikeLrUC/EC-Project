import random as rd

from .population import Population

class Selection:
    @staticmethod
    def tournament(tournament_size):
        def select(population: Population):
            return Population([Population(rd.sample(population.population, tournament_size)).best() for _ in range(population.size)])
        return select