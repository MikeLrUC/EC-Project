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
        self.best = None
        self.best_generation = None

    def run(self, generations):
        # TODO: acrescentar variaveis para guardar a geracao onde foi encontrado o best por cada run. Depois precisamos desta informacao guardada num ficheiro para podermos analisar mais tarde com os testes estatísticos.
        for i in range(generations):
            print(f"Generation: {i + 1} of {generations}")                              # Logging Info

            parents = self.population.select(self.selection_op)                         # Select Parents
            offspring = parents.breed(self.crossover_op, self.mutation_op, self.domain) # Breed (Crossover + Mutation)
            offspring.evaluate(self.fitness_func)                                       # Evaluate Offspring
            self.population = self.population.survive(self.survival_op, offspring)      # Select Survivors
            curr_gen_best = self.population.best()
            # print(f'best: {self.population.best()}')
            print(f'best: {curr_gen_best}')
            self.generations.append(deepcopy(self.population))                          # Logging Info

            # update best stats
            if self.best is None or self.best.fitness > curr_gen_best.fitness:          # if better, update
                self.best = curr_gen_best
                self.best_generation = i

        print(self.best)                                                                # Logging Info

if __name__ == "__main__":
    # Parameters
    N_POPULATION = 20
    SIZE_CHROMOSOME = 10
    N_GENERATIONS = 500
    N_RUNS = 1
    LEARNING_RATE = 0.1
    DOMAIN = [[-5.12, 5.12]] * SIZE_CHROMOSOME  # domain for the sphere function

    filenames = ["default_", "sa_"]

    default_ga_example_data = {
        "population" : Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, DOMAIN, Individual), 
        "crossover"  : Crossover.n_point_crossover(2),
        "mutation"   : Mutation.default(0.5),           # Mutation for the Default GA
        "selection"  : Selection.tournament(3),
        "survival"   : Survival.elitism(1),
        "fitness"    : Fitness.sphere,                  # Sphere as the target function 
        "domain"     : DOMAIN                           # domain for each gene of the individual
    }

    SA_ga_example_data = {
        "population" : Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, DOMAIN, Individual_SA), 
        "crossover"  : Crossover.n_point_crossover(2),
        "mutation"   : Mutation.SA(0.5, LEARNING_RATE),  # Mutation for the SA GA
        "selection"  : Selection.tournament(3),
        "survival"   : Survival.elitism(1),
        "fitness"    : Fitness.sphere,                   # Sphere as the target function 
        "domain"     : DOMAIN                            # Domain for each gene of the individual
    }

    # Running
    results = []
    figures = []
    labels = ["Default Algorithm", "SA Algorithm"]
    algorithms = ["GeneticAlgorithm(**default_ga_example_data)", "GeneticAlgorithm(**SA_ga_example_data)"]
    for e, filename in enumerate(filenames):
        runs = []
        for i in range(N_RUNS):
            ga: GeneticAlgorithm = eval(algorithms[e])
            ga.run(N_GENERATIONS)
            runs.append(ga.generations)
    
        # Logging    
        df = Logger.save_csv(runs, f"{filename}{N_GENERATIONS}")
        results.append(df)
    
        # Plotting
        figures.append(Plotter.simple_fitness(df, labels[e], maximize=False))
        figures.append(Plotter.fancy_fitness(df, labels[e]))

    figures += Plotter.box_plot(N_GENERATIONS, [results[0], results[1]], labels=labels, maximize=False)
    print(f'figures: {figures}')
    Logger.save_figures(figures, "Last_run")