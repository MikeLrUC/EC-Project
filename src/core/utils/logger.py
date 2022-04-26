import json
import pandas as pd

class Logger:
    DIR = __file__.split("core")[0] + "data/log/"

    @classmethod
    def save_csv(cls, filename, runs: list):
        d = dict()
        with open(cls.DIR + filename + ".csv", "w") as f:
            f.write("Run,Generation,Individual,Fitness\n")
            for r, run in enumerate(runs):
                for g, generation in enumerate(run):
                    population = generation.population
                    for i, individual in enumerate(population):
                        f.write(f"{r},{g},{i},{individual.fitness}\n")
        return pd.read_csv(cls.DIR + filename + ".csv")

    