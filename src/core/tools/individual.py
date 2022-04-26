class Individual:

    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

    def __str__(self):
        return f"{self.fitness} | {str(self.chromosome)}"

    def mutate(self, mutation_function):
        mutation_function(self)
        self.fitness = None
        return self

    def evaluate(self, fitness_function):
        self.fitness = fitness_function(self.chromosome)
        return self