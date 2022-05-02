from struct import unpack
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

class Plotter:

    def close():
        plt.close('all')

    def simple_fitness(runs: pd.DataFrame, algorithm, maximize=False, show=True):
        generations = runs["Generation"].unique()
        data_by_gen = runs[["Generation", "Fitness"]].groupby("Generation")

        best_results = data_by_gen.max() if maximize else data_by_gen.min()
        mean = data_by_gen.mean()

        # Plotting
        fig = plt.figure()
        
        # Average Fitness by Generations (All runs)
        plt.plot(generations, mean, color="r", label="Average")
        
        # Best Fitness by Generations (All runs)
        plt.plot(generations, best_results, color="g", label="Generation Best")
        
        # Text
        plt.title(f"[{algorithm}] Fitness Over Generations ({1 + runs['Run'].max()} runs)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show(block=show)
        return fig

    def fancy_fitness(runs: pd.DataFrame, algorithm, show=True):
        generations = runs["Generation"].unique()
        data_by_gen = runs[["Generation", "Fitness"]].groupby("Generation")

        maximum = data_by_gen.max()
        minimum = data_by_gen.min()
        median = data_by_gen.median()
        mean = data_by_gen.mean()
        std = data_by_gen.std()
        
        # Plotting
        fig = plt.figure()

        # Fitness Median by Generations (All runs)
        plt.plot(generations, median, color="blue", label="Median")
        
        # Minimum Fitness by Generations (All runs)
        plt.plot(generations, minimum, color="purple", label="Minimum")
        
        # Fitness Mean +- Std by Generations (All runs)
        plt.fill_between(generations, (mean - std)["Fitness"], (mean + std)["Fitness"], alpha=0.5 ,color="yellow", label="Standard Deviation")
        
        # Fitness Mean by Generations (All runs)
        plt.plot(generations, mean, color="red", label="Average")
        
        # Maximum Fitness by Generations (All runs)
        plt.plot(generations, maximum, color="green", label="Maximum")
        
        # Text
        plt.title(f"[{algorithm}] Fitness Over Generations ({1 + runs['Run'].max()} runs)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show(block=show)
        return fig

    def box_plot(df: pd.DataFrame, title, show=True):
        fig = plt.figure()
        plt.boxplot(df, labels=df.columns)
        plt.scatter([1] * len(df[df.columns[0]]), df[df.columns[0]])
        plt.scatter([2] * len(df[df.columns[1]]), df[df.columns[1]])
        plt.suptitle(title)
        plt.show(block=show)
        return fig

    def histogram(data: list, title, bins=25, normal=False, show=True):
        
        fig = plt.figure()
        
        plt.hist(data, bins=bins, density=normal)

        if normal:
            mu, std = st.norm.fit(data)
            x = np.linspace(*plt.xlim(), 1000)
            plt.plot(x, st.norm.pdf(x, mu, std))
        
        plt.title(f"{title} Histogram")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.show(block=show)
        return fig 



