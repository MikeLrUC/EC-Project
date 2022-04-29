from .population import Population

class Survival:

    def elitism(elitism_size):
        def survive(parents: Population, offspring: Population):
            parents.sort(reverse=False)
            offspring.sort(reverse=False)
            return Population(parents.population[:elitism_size] + offspring.population[: parents.size - elitism_size])
        return survive
