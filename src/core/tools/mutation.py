import random as rd

from .individual import Individual

class Mutation:

    @staticmethod
    def dummy(probability):
        def mutate(individual: Individual):
            if rd.random() <= probability:
                individual.chromosome[int(rd.random() * len(individual.chromosome))] = rd.random()
            return individual
        return mutate