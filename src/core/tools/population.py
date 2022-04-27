from .individual import Individual
from copy import deepcopy

class Population:

    def __init__(self, population : list[Individual]):
        self.population = population
        self.size = len(self.population)

    def __str__(self):
        string = ""
        for individual in self.population:
            string += f"{individual.__str__()}\n"
        return string

    def sort(self, reverse=False):
        self.population.sort(key=lambda individual : individual.fitness, reverse=reverse)

    def best(self) -> Individual:
        # FIXME isto tem de ser trocado. Nao pode ser reverse = True, pois queremos minimizar. Assim sendo os melhores sao os que tem menor valor de fitness
        # return sorted(self.population, reverse=True, key=lambda x : x.fitness)[0]
        return sorted(self.population, reverse=False, key=lambda x : x.fitness)[0]
    
    def evaluate(self, fitness_function):
        for individual in self.population:
            individual.evaluate(fitness_function)

    def avg(self):
        return sum([individual.fitness for individual in self.population]) / len(self.population)
    
    def select(self, selection_operator) -> 'Population':
        return selection_operator(self)

    def breed(self, crossover_operator, mutation_operator, domain: list) -> 'Population':
        #FIXME: Only works if the genes themselves are not mutable
        # TODO talvez usar o deepcopy antes para copiar os individuos para a criacao dos offsprings. Assim:
        pop_copy = [deepcopy(ind) for ind in self.population]
        offspring = Population(pop_copy)
        # offspring = Population(self.population[:]) 

        # Crossover
        for i in range(0, self.size - 1, 2):
            # FIXME Penso que aqui estava mal, estavas a usar a self.population em vez dos individuos do tournment selection
            # offspring.population[i], offspring.population[i + 1] = \
            #     crossover_operator(self.population[i], self.population[i + 1])
            offspring.population[i], offspring.population[i + 1] = \
                crossover_operator(offspring.population[i], offspring.population[i + 1])

        # Mutation
        for individual in offspring.population:
            mutation_operator(individual, domain)
        
        return offspring

    def survive(self, survival_operator, offspring) -> 'Population':
        return survival_operator(self, offspring)