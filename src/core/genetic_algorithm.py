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
    # Parameters
    N_POPULATION = 20                               #FIXME: Talvez reduzir para 10 dimensões?
    SIZE_CHROMOSOME = 10
    N_GENERATIONS = 500
    N_RUNS = 30
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

    # Genetic Algorithms related Info
    filenames = ["default", "sa"]
    labels = ["Default Algorithm", "SA Algorithm"]
    algorithms = ["GeneticAlgorithm(**default_ga_example_data)", "GeneticAlgorithm(**SA_ga_example_data)"]
    benchmarks = [Fitness.sphere, Fitness.rosenbrock, Fitness.step]
    
    # Clear Report File
    open(Logger.REPORT_FILE, "w").close()

    for benchmark in benchmarks:

        # Benchmark Problem
        problem = benchmark.__name__
        print(f"Problem: {problem}")
        
        # Set Fitness Functions for problem at hand
        default_ga_example_data["fitness"] = benchmark
        SA_ga_example_data["fitness"] = benchmark

        # Figures
        figures = []

        # Data lists
        results_all, results_best = [], []
        
        for e, filename in enumerate(filenames):    # Algorithms
            runs_generation, runs_best = [], []
            print("\nAlgorithm: ", labels[e])
            for i in range(N_RUNS):
                print("\t- Run: ", i, end="")
                ga: GeneticAlgorithm = eval(algorithms[e])
                ga.run(N_GENERATIONS)
                runs_generation.append(ga.generations)
                runs_best.append([ga.best_generation, ga.best])
                print()
        
            # Logging All
            df_all = Logger.save_all_csv(runs_generation, Logger.CSV_FILE(problem, filename, "all"))
            results_all.append(df_all)

            # Logging Best
            df_best = Logger.save_best_csv(runs_best, Logger.CSV_FILE(problem, filename, "best"))
            results_best.append(df_best)

            # Plotting Fitness Over Generations
            figures.append(Plotter.simple_fitness(df_all, labels[e], maximize=False, show=False))
            figures.append(Plotter.fancy_fitness(df_all, labels[e], show=False))


        #-*-# Statistics #-*-#
        select_data = lambda df, feature: df[0][[feature]].join(df[1][[feature]], lsuffix=f"_{filenames[0]}", rsuffix=f"_{filenames[1]}")

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
        figures.append(Plotter.box_plot(fitness_data, f"Best's Fitness Comparison ({len(fitness_data)} runs)", show=False))
        figures.append(Plotter.box_plot(generation_data, f"Best's Generation Comparison ({len(generation_data)} runs)", show=False))

        Logger.save_figures(figures, Logger.FIGURES_FILE(problem))
        
        # Close Figures because of Memory
        Plotter.close()

