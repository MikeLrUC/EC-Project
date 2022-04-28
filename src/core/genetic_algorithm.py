from tools import *
from utils import *
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, population: Population, crossover, mutation, selection, survival, fitness, domain):
        self.population = population
        self.crossover_op = crossover
        self.mutation_op = mutation
        self.selection_op = selection
        self.survival_op = survival
        self.fitness_func = fitness
        self.population.evaluate(self.fitness_func)
        self.generations = [deepcopy(self.population)]
        self.domain = domain # list with the following structure [[min_gene_1, max_gene_1], [min_gene_2, max_gene_2], ..., [min_gene_n, max_gene_n]]

    def run(self, generations):
        for i in range(generations):
            print(f"Generation: {i + 1} of {generations}")                              # Logging Info

            parents = self.population.select(self.selection_op)                         # Select Parents
            offspring = parents.breed(self.crossover_op, self.mutation_op, self.domain) # Breed (Crossover + Mutation)
            offspring.evaluate(self.fitness_func)                                       # Evaluate Offspring
            self.population = self.population.survive(self.survival_op, offspring)      # Select Survivors
            print(f'best: {self.population.best()}')
            self.generations.append(deepcopy(self.population))                          # Logging Info

        print(self.population.best())                                                   # Logging Info

if __name__ == "__main__":
    # Parameters
    N_POPULATION = 20
    SIZE_CHROMOSOME = 10
    N_GENERATIONS = 500
    N_RUNS = 10
    
    domain = [[-5.12, 5.12]] * SIZE_CHROMOSOME # domain for the sphere function
    learning_rate = 0.1

    filename="ga_run_"
    ilename="teste"
    
    data = {
        "population" : Generator.random_binary_population(SIZE_CHROMOSOME, N_POPULATION),
        "crossover"  : Crossover.dummy(0),
        "mutation"   : Mutation.dummy(0.05),
        "selection"  : Selection.tournament(2),
        "survival"   : Survival.elitism(1),
        "fitness"    : lambda x, genes_size : sum(x),
        "domain"     : []
    }

    normal_ga_example_data = {
        "population" : Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, domain, Individual), # create the population for the (default) normal ga
        "crossover"  : Crossover.n_point_crossover(2), # 2 point crossover
        "mutation"   : Mutation.mutate_default(0.5), # mutation for the nornal ga
        "selection"  : Selection.tournament(3),
        "survival"   : Survival.elitism(1),
        "fitness"    : Fitness.sphere, # sphere as the target function 
        "domain"     : domain # domain for each gene of the individual
    }

    SA_ga_example_data = {
        "population" : Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, domain, Individual_SA), # create the population for the (default) normal ga
        "crossover"  : Crossover.n_point_crossover(2), # 2 point crossover
        "mutation"   : Mutation.mutate_SA(0.5, learning_rate), # mutation for the SA ga
        "selection"  : Selection.tournament(3),
        "survival"   : Survival.elitism(1),
        "fitness"    : Fitness.sphere, # sphere as the target function 
        "domain"     : domain # domain for each gene of the individual
    }

    # Running
    runs = []
    for i in range(N_RUNS):
        # ga = GeneticAlgorithm(**data)
        ga = GeneticAlgorithm(**normal_ga_example_data)
        # ga = GeneticAlgorithm(**SA_ga_example_data)
        ga.run(N_GENERATIONS)
        runs.append(ga.generations)
    
    # Logging    
    df = Logger.save_csv(f"{filename}{N_GENERATIONS}", runs)
    
    # Plotting
    Plotter.simple_fitness(df)
    Plotter.fancy_fitness(df)
    Plotter.box_plot(N_GENERATIONS, [df, df], labels=["First Algorithm", "Second Algorithm"])