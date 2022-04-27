from .population import Population

class Survival:

    def elitism(elitism_size):
        def survive(parents: Population, offspring: Population):
            # FIXME nao pode ser reverse = True, pois e um problema de otimizacao. Assim sendo, o melhor tem o menor valor de fitness.
            # parents.sort(reverse=True)
            # offspring.sort(reverse=True)
            parents.sort(reverse=False)
            offspring.sort(reverse=False)
            return Population(parents.population[:elitism_size] + offspring.population[: parents.size - elitism_size])
        return survive
