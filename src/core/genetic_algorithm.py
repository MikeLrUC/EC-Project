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
        with open(Logger.LOG + "report.txt", "a") as f: 
            for i in range(generations):
                # Genetic Algorithm Core
                parents = self.population.select(self.selection_op)                             # Select Parents
                offspring = parents.breed(self.crossover_op, self.mutation_op, self.domain)     # Breed (Crossover + Mutation)
                offspring.evaluate(self.fitness_func)                                           # Evaluate Offspring
                self.population = self.population.survive(self.survival_op, offspring)          # Select Survivors

                # Update best
                current_generation_best = self.population.best()
                if self.best is None or self.best.fitness > current_generation_best.fitness:
                    self.best = current_generation_best
                    self.best_generation = i
                
                # Logging
                print(f"\r\t\t\t- Generation: {i + 1} of {generations}", end="")
                Logger.report(f"Generation: {i + 1} of {generations}", f)
                Logger.report(f'Run best:\n{self.best}', f)
                Logger.report(f'Generation best:\n{current_generation_best}', f)
                
                self.generations.append(deepcopy(self.population)), f

            Logger.report(self.best, f)

if __name__ == "__main__":
    # Parameters
    N_POPULATION = 20
    SIZE_CHROMOSOME = 10
    N_GENERATIONS = 500
    N_RUNS = 3
    LEARNING_RATE = 0.1
    DOMAIN = [[-5.12, 5.12]] * SIZE_CHROMOSOME      # Domain for the sphere function


    default_ga_example_data = {
        "population" : Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, DOMAIN, Individual), 
        "crossover"  : Crossover.n_point_crossover(2),
        "mutation"   : Mutation.default(0.5),           # Mutation for the Default GA
        "selection"  : Selection.tournament(3),
        "survival"   : Survival.elitism(1),
        "fitness"    : Fitness.sphere,                  # Sphere as the target function 
        "domain"     : DOMAIN                           # Domain for each gene of the individual
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

    #-*-# Running Algorithms #-*-#

    figures = []
    results_fitness, results_best = [], []

    # Genetic Algorithms related Info
    filenames = ["default", "sa"]
    labels = ["Default Algorithm", "SA Algorithm"]
    algorithms = ["GeneticAlgorithm(**default_ga_example_data)", "GeneticAlgorithm(**SA_ga_example_data)"]

    
    # Clear Report File
    open(Logger.LOG + "report.txt", "w").close()
    
    print("Comparing: ")
    for e, filename in enumerate(filenames):
        runs_generation, runs_best = [], []
        print("\nAlgorithm: ", labels[e])
        for i in range(N_RUNS):
            print("\t- Run: ", i, end="")
            ga: GeneticAlgorithm = eval(algorithms[e])
            ga.run(N_GENERATIONS)
            runs_generation.append(ga.generations)
            runs_best.append([ga.best_generation, ga.best])
            print()
    
        # Logging Fitness
        df_fitness = Logger.save_fitness_csv(runs_generation, f"{filename}_fitness")
        results_fitness.append(df_fitness)

        # Logging Best
        df_best = Logger.save_best_csv(runs_best, f"{filename}_best")
        results_best.append(df_best)

        # Plotting Fitness
        figures.append(Plotter.simple_fitness(df_fitness, labels[e], maximize=False, show=False))
        figures.append(Plotter.fancy_fitness(df_fitness, labels[e], show=False))

    # Save Fitness Figures
    figures += Plotter.box_plot(N_GENERATIONS, [results_fitness[0], results_fitness[1]], labels=labels, maximize=False, show=False)
    Logger.save_figures(figures, "Last_run")


    #-*-# Statistics #-*-#

    #FIXME: Probably doesnt use every run at once to make the statistics. Have to check 
    fitness_data = results_fitness[0][["Fitness"]].join(results_fitness[1][["Fitness"]], lsuffix="_default", rsuffix="_sa")
    Statistics.analyse(fitness_data, True, "Normality Hists")
    
    print("\nDone: Statistical Analysis\n")
    print("The runs report is found at report.txt")
    print("The data statistics is found at statistics.txt")
