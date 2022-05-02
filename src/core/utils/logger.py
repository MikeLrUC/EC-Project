import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

class Logger:
    DIR = __file__.split("core")[0] + "data/"
    LOG = DIR + "log/"
    OTHERS = DIR + "others/"
    REPORT_FILE = OTHERS + "report.txt"
    CSV_FILE = lambda problem, algorithm, dtype: f"{problem}_{algorithm}_{dtype}"
    HIST_FILE = lambda problem, dtype: f"{problem}_histograms_{dtype}"
    STATISTICS_FILE = lambda problem, dtype: f"{problem}_statistics_{dtype}.txt"
    FIGURES_FILE = lambda problem: f"{problem}_figures" 

    @classmethod
    def save_all_csv(cls, runs: list, filename):
        with open(cls.LOG + filename + ".csv", "w") as f:
            f.write("Run,Generation,Individual,Fitness\n")
            for r, run in enumerate(runs):
                for g, generation in enumerate(run):
                    population = generation.population
                    for i, individual in enumerate(population):
                        f.write(f"{r},{g},{i},{individual.fitness}\n")
        return pd.read_csv(cls.LOG + filename + ".csv")

    @classmethod
    def save_best_csv(cls, runs: list, filename):
        with open(cls.LOG + filename + ".csv", "w") as f:
            f.write("Run,Generation,Fitness,Chromosome\n")
            for r, (best_gen, best) in enumerate(runs):
                f.write(f"{r},{best_gen},{best.fitness},\"{best.chromosome}\"\n")
        return pd.read_csv(cls.LOG + filename + ".csv")

    @classmethod
    def save_figures(cls, figures, filename):
        with PdfPages(cls.OTHERS + filename + ".pdf") as pdf:
            for i in range(0, len(figures)):
                pdf.savefig(figures[i])

    def report(text, filepointer):
        print(text, file=filepointer)
    