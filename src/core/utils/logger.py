import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

class Logger:
    DIR = __file__.split("core")[0] + "data/"
    LOG = DIR + "log/"
    OTHERS = DIR + "others/"

    @classmethod
    def save_csv(cls, runs: list, filename):
        with open(cls.LOG + filename + ".csv", "w") as f:
            f.write("Run,Generation,Individual,Fitness\n")
            for r, run in enumerate(runs):
                for g, generation in enumerate(run):
                    population = generation.population
                    for i, individual in enumerate(population):
                        f.write(f"{r},{g},{i},{individual.fitness}\n")
        return pd.read_csv(cls.LOG + filename + ".csv")

    @classmethod
    def save_figures(cls, figures, filename):
        with PdfPages(cls.OTHERS + filename + ".pdf") as pdf:
            for i in range(0, len(figures)):
                pdf.savefig(figures[i])
    