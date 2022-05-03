from math import floor, cos, pi, sin, sqrt, prod
import numpy as np

from .individual import *


class Fitness:
    def sphere(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [0, 0, ..., 0] == 0
        domain -> [-5.12, 5.12] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = sum([gene**2 for gene in genes])
        return fitness

    def rosenbrock(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [1, 1, ..., 1] == 0
        domain ->  [-2.048, 2.048] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = sum([(1 - genes[i])**2 + 100 * (genes[i + 1] - genes[i]**2)**2 for i in range(len(genes) - 1)])
        return fitness

    def step(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [-5.12, -5.12, ..., -5.12] == 0
        domain -> [-5.12, 5.12] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = 6 * genes_size + sum([floor(gene) for gene in genes])
        return fitness
    
    def quartic(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [0, 0, ..., 0] == 0
        domain -> [-1.28, 1.28] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = sum([i * genes[i]**4 for i in range(genes_size)]) + np.random.normal(0, 1)
        return fitness
    
    def rastrigin(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [0, 0, ..., 0] == 0
        domain -> [-5.12, 5.12] * genes_size
        '''
        A = 10 # A usually equal to 10.
        genes = chromosome[: genes_size]
        fitness = A * genes_size + sum([gene**2 - A * cos(2 * pi * gene) for gene in genes])
        return fitness
    
    def schwefel(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [420.9687, ..., 420.9687] == -genes_size * -418.9829
        domain -> [-500, 500] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = sum([-gene * sin(sqrt(abs(gene))) for gene in genes])
        return fitness

    def griewangk(chromosome, genes_size):
        '''
        chromosome: list with the genes or genes + stds.
        genes_size: int with the size of the genes (so that we know how many values in the chromosome are genes and not stds)
        min -> [0, 0, ..., 0] == 0
        domain -> [-600, 600] * genes_size
        '''
        genes = chromosome[: genes_size]
        fitness = 1 + (1 / 4000) * sum([gene**2 for gene in genes]) + prod([cos(genes[i] / sqrt(i)) for i in range(genes_size)])
        return fitness