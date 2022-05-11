import random as rd

from .individual import Individual, Individual_SA
from .generator import Generator

class Mutation:
    
    # Per gene mutation
    def default(probability):
        '''
        probability: probability value to be used in the per gene mutation.
        returns a function that will perform the mutation.
        '''
        def mutate(individual: Individual, domain):
            '''
            individual: the individual that we want to mutate.
            domain: list with the intervals where each gene should belong to.
            '''
            # if the mutation is to be done in gene i, then a random value on the possible interval is chosen.
            individual.chromosome = [Generator.pick_value_in_interval(domain[i]) if rd.random() < probability else individual.chromosome[i] for i in range(len(individual.chromosome))]
            return individual
        return mutate
    
    # Per gene mutation Self-Adaptation in Evolutionary Strategies
    def SA(probability, learning_rate):
        '''
        probability: probability value to be used in the per gene mutation.
        learning_rate: coeficient used to control the change in the current standard deviation.
        returns a function that will perform the mutation.
        '''
        def mutate(individual: Individual_SA, domain):
            '''
            individual: the individual that we want to mutate.
            domain: list with the intervals where each gene should belong to.
            '''
            # Mutate stds
            new_stds = [Generator.new_std_SA(individual.chromosome[i + individual.genes_size], learning_rate) for i in range(individual.genes_size)]

            # Mutate genes
            new_genes = [min(max(Generator.new_gene_SA(individual.chromosome[i], new_stds[i]), domain[i][0]), domain[i][1]) for i in range(individual.genes_size)]


            # Build Chromosome
            individual.chromosome = new_genes + new_stds

            return individual
        return mutate

if __name__ == '__main__':
    size = 20
    domain = [[0, 1]] * size
    
    # default ga
    m = Mutation.default(0.5)
    p1 = Individual([1] * size, None)
    new_p1 = m(p1, domain)
    print(f'p1: {new_p1}\n')

    # ga with SA
    learning_rate = 0.1
    m_sa = Mutation.SA(0.5, learning_rate)
    p2 = Individual_SA([1] * size, None)
    new_p2 = m_sa(p2, domain, learning_rate)
    print(f'p2: {new_p2}\n')
