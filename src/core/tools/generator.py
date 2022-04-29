import random as rd
import numpy as np

from .population import Population

class Generator:

    def random_float_generation(chromosome_size, population_size, domain, Individual_constructor):
        '''
        chromosome_size: size of the genotype of the individual.
        population_size: size of the population.
        domain: list with the intervals where each gene should belong to.
        Individual_constructor: constructor for the type of individual that we want.
        '''
        return Population( [Individual_constructor( [Generator.pick_value_in_interval(domain[i]) for i in range(chromosome_size)], None ) for _ in range(population_size)] )
    
    def pick_value_in_interval(interval):
        '''
        interval: list with the minimum and maximum value for some gene.
        '''
        return (interval[1] - interval[0]) * rd.random() + interval[0]
    
    #TODO: Miguel review this
    def new_gene_SA(old_gene, std):
        '''
        old_gene: the value that the current gene has.
        std: standard deviation associated with the current gene.
        '''
        return old_gene + std * np.random.normal(0, 1)
    
    def new_std_SA(old_std, learning_rate):
        '''
        old_std: the value of the current standard deviation.
        learning_rate: coeficient used to control the change in the current standard deviation.
        '''
        return old_std * np.exp(learning_rate * np.random.normal(0, 1))