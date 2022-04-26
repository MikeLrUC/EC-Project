from tools import *
from utils import *
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, population: Population, crossover, mutation, selection, survival, fitness):
        self.population = population
        self.crossover_op = crossover
        self.mutation_op = mutation
        self.selection_op = selection
        self.survival_op = survival
        self.fitness_func = fitness
        self.population.evaluate(self.fitness_func)
        self.generations = [deepcopy(self.population)]

    def run(self, generations):
        for i in range(generations):
            print(f"Generation: {i + 1} of {generations}")                              # Logging Info

            parents = self.population.select(self.selection_op)                         # Select Parents
            offspring = parents.breed(self.crossover_op, self.mutation_op)              # Breed (Crossover + Mutation)
            offspring.evaluate(self.fitness_func)                                       # Evaluate Offspring
            self.population = self.population.survive(self.survival_op, offspring)      # Select Survivors
            
            self.generations.append(deepcopy(self.population))                          # Logging Info

        print(self.population.best())                                                   # Logging Info

if __name__ == "__main__":
    # Parameters
    N_POPULATION = 20
    SIZE_CHROMOSOME = 10
    N_GENERATIONS = 1000
    N_RUNS = 100
    
    filename="ga_run_"
    
    data = {
        "population" : Generator.random_binary_population(SIZE_CHROMOSOME, N_POPULATION),
        "crossover"  : Crossover.dummy(0),
        "mutation"   : Mutation.dummy(0.05),
        "selection"  : Selection.tournament(2),
        "survival"   : Survival.elitism(1),
        "fitness"    : lambda x : sum(x)
    }

    # Running
    runs = []
    for i in range(N_RUNS):
        ga = GeneticAlgorithm(**data)
        ga.run(N_GENERATIONS)
        runs.append(ga.generations)
    
    # Logging    
    df = Logger.save_csv(f"{filename}{N_GENERATIONS}", runs)
    
    # Plotting
    Plotter.simple_fitness(df)
    Plotter.fancy_fitness(df)
    Plotter.box_plot(N_GENERATIONS, [df, df], labels=["First Algorithm", "Second Algorithm"])