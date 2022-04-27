from operator import le
import random as rd
import numpy as np

from .individual import Individual, Individual_SA
from .generator import Generator

class Mutation:

    def dummy(probability):
        def mutate(individual: Individual, domain = None):
            if rd.random() <= probability:
                individual.chromosome[int(rd.random() * len(individual.chromosome))] = rd.random()
            return individual
        return mutate
    
    # per gene mutation
    def mutate_default(probability):
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
    
    # per gene mutation Self-Adaptation in Evolutionary Strategies
    def mutate_SA(probability):
        '''
        probability: probability value to be used in the per gene mutation.
        returns a function that will perform the mutation.
        '''
        def mutate(individual: Individual_SA, domain, learning_rate):
            '''
            individual: the individual that we want to mutate.
            domain: list with the intervals where each gene should belong to.
            learning_rate: coeficient used to control the change in the current standard deviation.
            '''
            # mutate genes
            new_genes = [min(max(Generator.new_gene_SA(individual.chromosome[i], individual.chromosome[i + individual.genes_size]), domain[i][0]), domain[i][1]) for i in range(individual.genes_size)]

            # mutate stds
            new_stds = [Generator.new_std_SA(individual.chromosome[i + individual.genes_size], learning_rate) for i in range(individual.genes_size)]

            # build chromosome
            individual.chromosome = new_genes + new_stds

            return individual
        return mutate

if __name__ == '__main__':
    size = 20
    domain = [[0, 1]] * size
    
    # default ga
    m = Mutation.mutate_default(0.5)
    p1 = Individual([1] * size, None)
    new_p1 = m(p1, domain)
    print(f'p1: {new_p1}\n')

    # ga with SA
    m_sa = Mutation.mutate_SA(0.5)
    learning_rate = 0.1
    p2 = Individual_SA([1] * size, None)
    new_p2 = m_sa(p2, domain, learning_rate)
    print(f'p2: {new_p2}\n')
