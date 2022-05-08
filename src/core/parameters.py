import numpy as np
import random as rd
import matplotlib.pyplot as plt
from copy import deepcopy

from tools import *
from utils import *
from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    # Clear Report File
    open(Logger.REPORT_FILE, "w").close()


    #==#==#==#==# SCRIPT PARAMETERS TO TWEAK #==#==#==#==#

    # Parameters to Test
    parameters = {
        "Mutation Probability": [0.1, 0.3, 0.5],
        "Population": [10, 20, 30],
        "Learning Rate": [0.1, 0.5, 0.9],
        "Std Domain": [[0,1], [1,1], [-1, 1]]
    }

    # Parameters Fixed, while testing some
    fixed_parameters = {
        "Mutation Probability": 0.1,
        "Population": 10,
        "Learning Rate": 0.1,
        "N Point Crossover": 2,
        "Tournament Size": 3,
        "Elitism Size": 1,
        "Std Domain": [0, 1]
    }

    # Run Seed Generator Seed
    rd.seed(42)
    
    # Configs
    N_RUNS = 30
    N_GENERATIONS = 1
    SIZE_CHROMOSOME = 20
    SEEDS = [rd.randint(0, 10000) for _ in range(N_RUNS)]   

    benchmarks = [
        [Fitness.sphere, [-5.12, 5.12]],
        [Fitness.rosenbrock, [-2.048, 2.048]],
        [Fitness.step, [-5.12, 5.12]],
        [Fitness.quartic, [-1.28, 1.28]], 
        [Fitness.rastrigin, [-5.12, 5.12]],
        #[Fitness.schwefel, [-500, 500]], # Commented bc the target fitness is -XXXX .... that messes up the scale to the others
        [Fitness.griewangk, [-600, 600]] 
    ]
    
    # Plotting Info
    colors = ["green", "orange", "red", "blue", "pink", "black", "yellow"]
    
    plot_3d_view_angle = {
        "elev": 15,
        "azim": -40
    }

    plot_legend_properties = {
        "loc":9,
        "prop": {
            "size": 8
        },
        "ncol": 3
    }
    #==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#


    # For each parameter
    figures = []
    for param_name, values in parameters.items():
        # Create new figure with 2 subplots
        fig = plt.figure(figsize=(10,6))
        axs = [fig.add_subplot(1,2,1, projection="3d"), fig.add_subplot(1,2,2, projection="3d")]

        # For each benchmark Problem
        for p, (benchmark, domain) in enumerate(benchmarks):
            print(f"\n\nProblem {p + 1} of {len(benchmarks)}: {benchmark.__name__}")

            # Default Parameters (Fixed)
            default_params = deepcopy(fixed_parameters)

            # Plotting Data Structure
            problems_results = {
                bm.__name__ : {
                    algorithm : {
                        topic: {
                            "param_value":[],  "score":[]
                        } for topic in ["fitness", "generation"]
                    } for algorithm in ["default", "sa"]
                } for bm, dmn in benchmarks
            }

            # For possible values for the current testing parameter
            for v, value in enumerate(values):
                print(f"\n\nTesting: {param_name} [{value}]")
                
                # Overwrite default parameter
                default_params[param_name] = value

                # Default Algorithm Data
                default = {
                    "initializer": Generator.random_float_generation(SIZE_CHROMOSOME, default_params["Population"], [domain] * SIZE_CHROMOSOME, Individual, default_params["Std Domain"]), 
                    "crossover"  : Crossover.n_point_crossover(default_params["N Point Crossover"]),
                    "mutation"   : Mutation.default(default_params["Mutation Probability"]),           # Mutation for the Default GA
                    "selection"  : Selection.tournament(default_params["Tournament Size"]),
                    "survival"   : Survival.elitism(default_params["Elitism Size"]),
                    "fitness"    : benchmark,
                    "domain"     : [domain] * SIZE_CHROMOSOME
                }

                # Self-Adaptation Algorithm Data
                sa = {
                    "initializer": Generator.random_float_generation(SIZE_CHROMOSOME, default_params["Population"], [domain] * SIZE_CHROMOSOME, Individual_SA, default_params["Std Domain"]), 
                    "crossover"  : Crossover.n_point_crossover(default_params["N Point Crossover"]),
                    "mutation"   : Mutation.SA(default_params["Mutation Probability"], default_params["Learning Rate"]),  # Mutation for the SA GA
                    "selection"  : Selection.tournament(default_params["Tournament Size"]),
                    "survival"   : Survival.elitism(default_params["Elitism Size"]),
                    "fitness"    : benchmark,
                    "domain"     : [domain] * SIZE_CHROMOSOME
                } 
                
                algorithms = list(zip(["default", "sa"], [default, sa]))

                # For each algorithm
                for name, data in algorithms:
                    print(f"\nAlgorithm: {name}")
                    
                    # Run N_RUNS
                    runs_best = []
                    for i in range(N_RUNS):
                        print(f"\r\t- Run: {i + 1} of {N_RUNS}", end="")
                        
                        # Setting Seeds
                        rd.seed(SEEDS[i])
                        np.random.seed(SEEDS[i])

                        # Running Algorithm
                        ga = GeneticAlgorithm(**data)
                        ga.run(N_GENERATIONS)
                        runs_best.append([ga.best_generation, ga.best])
                    
                    # Pick Best
                    best = sorted(runs_best, key=lambda best: best[1].fitness)[0]
                    
                    problems_results[benchmark.__name__][name]["fitness"]["param_value"].append(v)
                    problems_results[benchmark.__name__][name]["fitness"]["score"].append(best[1].fitness) 

                    problems_results[benchmark.__name__][name]["generation"]["param_value"].append(v)
                    problems_results[benchmark.__name__][name]["generation"]["score"].append(best[0]) 
            
            for i, (name, data) in enumerate(algorithms):
                # Fitness Plot Lines
                plot = axs[0]
                x = problems_results[benchmark.__name__][name]["fitness"]["param_value"]
                y = [i] * len(problems_results[benchmark.__name__][name]["fitness"]["score"])
                z = problems_results[benchmark.__name__][name]["fitness"]["score"]
                plot.plot(x, y, z, "-x", color=colors[p], label=benchmark.__name__ if i else None)

                # Generation Plot Lines
                plot = axs[1]
                x = problems_results[benchmark.__name__][name]["generation"]["param_value"]
                y = [i] * len(problems_results[benchmark.__name__][name]["generation"]["score"])
                z = problems_results[benchmark.__name__][name]["generation"]["score"]
                plot.plot(x, y, z, "-x", color=colors[p], label=benchmark.__name__ if i else None)
        
        # Fitness Plot Labels
        plot = axs[0]
        plot.set_xlabel(param_name)
        plot.set_xticks([i for i in range(len(values))])
        plot.xaxis.set_ticklabels([str(value) for value in values])
        plot.set_ylabel("Algorithm")
        plot.set_yticks([0, 1])
        plot.yaxis.set_ticklabels(["Default", "SA"])
        plot.set_zlabel("Fitness")
        plot.view_init(**plot_3d_view_angle)
        plot.set_title(f"Bests' Fitness")
        plot.legend(**plot_legend_properties)

        # Generation Plot Labels
        plot = axs[1]
        plot.set_xlabel(param_name)
        plot.set_xticks([i for i in range(len(values))])
        plot.xaxis.set_ticklabels([str(value) for value in values])
        plot.set_ylabel("Algorithm")
        plot.set_yticks([0, 1])
        plot.yaxis.set_ticklabels(["Default", "SA"])
        plot.set_zlabel("Generations")
        plot.view_init(**plot_3d_view_angle)
        plot.set_title(f"Bests' Generations")
        plot.legend(**plot_legend_properties)

        plt.suptitle(f"{param_name} Parameter Testing\nBest of {N_GENERATIONS} generations of {N_RUNS} runs" )
        figures.append(fig)

    # Save Figures
    Logger.save_figures(figures, "parameters")





