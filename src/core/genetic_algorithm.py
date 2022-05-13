import random as rd
import numpy as np

from tools import *
from utils import *
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, initializer, crossover, mutation, selection, survival, fitness, domain):
        self.population: Population = initializer()
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
        with open(Logger.REPORT_FILE, "a") as f: 
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
    # Clear Report File
    open(Logger.REPORT_FILE, "w").close()

    # Run Seed Generator Seed
    rd.seed(42)

    # Default Parameters
    N_RUNS = 30
    N_GENERATIONS = 300
    N_POPULATION = 100                          
    SIZE_CHROMOSOME = 20
    LEARNING_RATE = 0.9
    STD_DOMAIN = [0, 1]
    SEEDS = [rd.randint(0, 10000) for _ in range(N_RUNS)]
    
    #-*-# Running Algorithms #-*-#

    benchmarks = [
        [Fitness.sphere, [-5.12, 5.12]],
        [Fitness.rosenbrock, [-2.048, 2.048]],
        [Fitness.step, [-5.12, 5.12]],
        [Fitness.quartic, [-1.28, 1.28]], 
        [Fitness.rastrigin, [-5.12, 5.12]],
        #[Fitness.schwefel, [-500, 500]], 
        #[Fitness.griewangk, [-600, 600]] 
    ]

    labels = ["Default Algorithm", "SA Algorithm"]

    for p, (benchmark, domain) in enumerate(benchmarks):

        # Benchmark Problem
        problem = benchmark.__name__
        print(f"Problem {p + 1} of {len(benchmarks)}: {problem}")
        
        # Algorithms
        default = {
            "initializer": Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, [domain] * SIZE_CHROMOSOME, Individual, STD_DOMAIN), 
            "crossover"  : Crossover.n_point_crossover(2, 0.8),
            "mutation"   : Mutation.default(0.1),           # Mutation for the Default GA
            "selection"  : Selection.tournament(3),
            "survival"   : Survival.elitism(1),
            "fitness"    : benchmark,
            "domain"     : [domain] * SIZE_CHROMOSOME
        }

        sa = {
            "initializer": Generator.random_float_generation(SIZE_CHROMOSOME, N_POPULATION, [domain] * SIZE_CHROMOSOME, Individual_SA, STD_DOMAIN), 
            "crossover"  : Crossover.n_point_crossover(2, 0.8),
            "mutation"   : Mutation.SA(0.1, LEARNING_RATE),  # Mutation for the SA GA
            "selection"  : Selection.tournament(3),
            "survival"   : Survival.elitism(1),
            "fitness"    : benchmark,
            "domain"     : [domain] * SIZE_CHROMOSOME
        } 

        algorithms = list(zip(["default", "sa"], [default, sa]))

        # Running Info Data lists
        figures, results_all, results_best = [], [], []
        
        for i, (name, data) in enumerate(algorithms): 
            print(f"\nAlgorithm: {name}")

            runs_generation, runs_best = [], []
            for j in range(N_RUNS):
                print(f"\r\t- Run: {j + 1} of {N_RUNS}", end="")

                # Setting Seeds
                rd.seed(SEEDS[j])
                np.random.seed(SEEDS[j])

                # Running GA
                ga = GeneticAlgorithm(**data)
                ga.run(N_GENERATIONS)
                runs_generation.append(ga.generations)
                runs_best.append([ga.best_generation, ga.best])
        
            # Logging All
            df_all = Logger.save_all_csv(runs_generation, Logger.CSV_FILE(problem, name, "all"))
            results_all.append(df_all)

            # Logging Best
            df_best = Logger.save_best_csv(runs_best, Logger.CSV_FILE(problem, name, "best"))
            results_best.append(df_best)

            # Plotting Fitness Over Generations
            figures.append(Plotter.simple_fitness(df_all, labels[i], maximize=False, show=False))
            figures.append(Plotter.fancy_fitness(df_all, labels[i], show=False))


        #-*-# Statistics #-*-#
        select_data = lambda results, feature: results[0][[feature]].join(results[1][[feature]], lsuffix=f"_{algorithms[0][0]}", rsuffix=f"_{algorithms[1][0]}")

        # Analyse Fitness Values where the best was found, for each one of the N_RUNS runs
        fitness_data = select_data(results_best, "Fitness")
        fitness_data.columns = [ f"[{label}] Fitness" for label in labels]
        Statistics.analyse(fitness_data, True, Logger.STATISTICS_FILE(problem, "fitness"), Logger.HIST_FILE(problem, "fitness"))

        # Analyse Generation Values where the best was found, for each one of the N_RUNS runs
        generation_data = select_data(results_best, "Generation")
        generation_data.columns = [ f"[{label}] Generation" for label in labels]
        Statistics.analyse(generation_data, True, Logger.STATISTICS_FILE(problem, "generation"), Logger.HIST_FILE(problem, "generation"))
        
        print("\nDone: Statistical Analysis\n")

        # Save Fitness and Generation Comparison Figures
        figures.append(Plotter.box_plot(fitness_data, f"Best's Fitness Comparison ({N_GENERATIONS} generations, {len(fitness_data)} runs)", show=False))
        figures.append(Plotter.box_plot(generation_data, f"Best's Generation Comparison ({N_GENERATIONS} generations, {len(generation_data)} runs)", show=False))

        Logger.save_figures(figures, Logger.FIGURES_FILE(problem))
        
        # Close Figures because of Memory
        Plotter.close()

