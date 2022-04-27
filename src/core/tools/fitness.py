from math import floor

from .individual import *


class Fitness:

    def sphere(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [0, 0, ..., 0]
        '''
        genes = chromosome[: genes_size]
        fitness = sum([gene**2 for gene in genes])
        return fitness

    def rosenbrock(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [1, 1, ..., 1]
        '''
        genes = chromosome[: genes_size]
        fitness = sum([(1 - genes[i])**2 + 100 * (genes[i + 1] - genes[i]**2)**2 for i in range(len(genes) - 1)])
        return fitness

    def step(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [-5.12, -5.12, ..., -5.12]
        '''
        genes = chromosome[: genes_size]
        fitness = 6 * genes_size + sum([floor(gene) for gene in genes])
        return fitness