class Individual:

    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness
        self.genes_size = len(chromosome)

    def __str__(self):
        return f"{self.fitness} | {str(self.chromosome)}"

    def mutate(self, mutation_function):
        mutation_function(self)
        self.fitness = None
        return self

    def evaluate(self, fitness_function):
        self.fitness = fitness_function(self.chromosome, self.genes_size)
        return self

class Individual_SA(Individual):
    def __init__(self, chromosome, fitness):
        # all genes start with std = 1, so we will have the distribution N(0,1)
        # This values can be changed during execution time
        stds = [1] * len(chromosome) # list with the standard deviations of the normal distributions of each gene
        self.chromosome = chromosome + stds # by combining the two, we can use the same crossover function for both algorithms
        self.fitness = fitness
        self.genes_size = len(chromosome)
