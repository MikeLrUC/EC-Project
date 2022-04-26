from .individual import Individual

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
        return sorted(self.population, reverse=True, key=lambda x : x.fitness)[0]
    
    def evaluate(self, fitness_function):
        for individual in self.population:
            individual.evaluate(fitness_function)

    def avg(self):
        return sum([individual.fitness for individual in self.population]) / len(self.population)
    
    def select(self, selection_operator) -> 'Population':
        return selection_operator(self)

    def breed(self, crossover_operator, mutation_operator) -> 'Population':
        #FIXME: Only works if the genes themselves are not mutable
        offspring = Population(self.population[:]) 

        # Crossover
        for i in range(0, self.size - 1, 2):
            offspring.population[i], offspring.population[i + 1] = \
                crossover_operator(self.population[i], self.population[i + 1])

        # Mutation
        for individual in offspring.population:
            mutation_operator(individual)
        
        return offspring

    def survive(self, survival_operator, offspring) -> 'Population':
        return survival_operator(self, offspring)