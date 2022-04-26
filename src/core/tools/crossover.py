import random as rd

from .individual import Individual

class Crossover:

    def dummy(probability):
        def crossover(p1: Individual, p2: Individual):
            if rd.random() <= probability:
                p1, p2 = p2, p1
            return p1, p2
        return crossover

if __name__ == "__main__":
    p1, p2 = Individual([1, 2, 3], None), Individual([4, 5, 6], None)
    c = Crossover.dummy(0.5)
    p1,p2 = c(p1,p2)
    print(p1)
    print(p2)

