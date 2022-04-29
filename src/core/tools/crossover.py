import random as rd

from .individual import Individual

class Crossover:

    # N point crossover
    def n_point_crossover(n):
        '''
        n: number of points in the n point crossover.
        returns a function that will perform the n point crossover
        '''
        def crossover(p1: Individual, p2: Individual):
            '''
            p1: parent number one.
            p2: parent number two.
            returns the two new individuals.
            '''
            # get n points
            chromosome_size = len(p1.chromosome)
            inxs = range(chromosome_size)
            points = [rd.choice(inxs) for _ in range(n)]
            points.sort()
            points = [0] + points + [chromosome_size]
            # perform crossover
            new_p1 = []
            new_p2 = []
            for i in range(1, len(points)):
                if i % 2 != 0:
                    new_p1 += p1.chromosome[points[i - 1] : points[i]]
                    new_p2 += p2.chromosome[points[i - 1] : points[i]]
                else:
                    new_p1 += p2.chromosome[points[i - 1] : points[i]]
                    new_p2 += p1.chromosome[points[i - 1] : points[i]]
            p1.chromosome, p2.chromosome = new_p1, new_p2

            return p1, p2
        return crossover

if __name__ == "__main__":
    # p1, p2 = Individual([1, 2, 3], None), Individual([4, 5, 6], None)
    # c = Crossover.dummy(0.5)
    # p1,p2 = c(p1,p2)
    # print(p1)
    # print(p2)

    p1, p2 = Individual([1] * 20, None), Individual([0] * 20, None)
    c = Crossover.n_point_crossover(3)
    new_p1, new_p2 = c(p1, p2)
    print(f'p1: {new_p1}\np2: {new_p2}\n')
