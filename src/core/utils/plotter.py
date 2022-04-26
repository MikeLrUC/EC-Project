import pandas as pd
import matplotlib.pyplot as plt

class Plotter:

    @staticmethod
    def simple_fitness(runs: pd.DataFrame):
        generations = runs["Generation"].unique()
        data_by_gen = runs[["Generation", "Fitness"]].groupby("Generation")
        maximum = data_by_gen.max()
        mean = data_by_gen.mean()
        plt.figure()
        
        # Average Fitness by Generations (All runs)
        plt.plot(generations, mean, color="r", label="Average")
        
        # Max Fitness by Generations (All runs)
        plt.plot(generations, maximum, color="g", label="Best")
        
        # Text
        plt.title(f"Fitness Over Generations ({1 + runs['Run'].max()} runs)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

    def fancy_fitness(runs: pd.DataFrame):
        generations = runs["Generation"].unique()
        data_by_gen = runs[["Generation", "Fitness"]].groupby("Generation")
        maximum = data_by_gen.max()
        minimum = data_by_gen.min()
        median = data_by_gen.median()
        mean = data_by_gen.mean()
        std = data_by_gen.std()
        
        plt.figure()

        # Fitness Median by Generations (All runs)
        plt.plot(generations, median, color="blue", label="Median")
        
        # Minimum Fitness by Generations (All runs)
        plt.plot(generations, minimum, color="purple", label="Minimum")
        
        # Fitness Mean +- Std by Generations (All runs)
        plt.plot(generations, mean + std, color="yellow", label="Standard Deviation")
        plt.plot(generations, mean - std, color="yellow")
        
        # Fitness Mean by Generations (All runs)
        plt.plot(generations, mean, color="red", label="Average")
        
        # Maximum Fitness by Generations (All runs)
        plt.plot(generations, maximum, color="green", label="Maximum")
        
        # Text
        plt.title(f"Fitness Over Generations ({1 + runs['Run'].max()} runs)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()


    @staticmethod
    def box_plot(generation, runs: list, labels=["Algorithm 0", "Algorithm 1"]):
        n_algorithms = len(runs)

        data, means, maxs = [], [], []
        for i, run in enumerate(runs):
            data.append(run[run["Generation"] == generation][["Run", "Fitness"]].groupby("Run"))
            means.append(data[i].mean())
            maxs.append(data[i].max())

        # Fitness Comparison of given Generation
        comparisons = [means, maxs]
        titles = ["Mean", "Maximum"]
        colors = ["red", "green"]
        for i, comparison in enumerate(comparisons):
            plt.subplots(1, n_algorithms)
            plt.suptitle(f"Generation {generation} Fitness {titles[i]} by {1 + runs[0]['Run'].max()} Runs")
            for j in range(n_algorithms):
                plt.subplot(1, n_algorithms, j+1)
                plt.scatter([1] * len(comparison[j]), comparison[j], color=colors[i])
                plt.boxplot(comparison[j], labels=[labels[j]])
        plt.show()



