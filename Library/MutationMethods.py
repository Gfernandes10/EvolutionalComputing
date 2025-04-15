import random

class MutationMethods:
    """
    A class containing various mutation methods for genetic algorithms.

    This class provides implementations of common mutation techniques used in genetic algorithms
    to introduce diversity into the population and avoid premature convergence.
    """

    @staticmethod
    def random_mutation_on_individual_genes(individual, mutation_rate):
        """
        Performs random mutation on individual genes.

        Parameters:
        - individual (list): The individual to mutate.
        - mutation_rate (float): The probability of mutating each gene.

        Returns:
        - list: The mutated individual.
        """
        mutated_individual = individual[:]
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                mutated_individual[i] = random.uniform(0, 1)
        return mutated_individual
    
    
    @staticmethod
    def bit_flip_mutation(individual, mutation_rate):
        """
        Performs bit-flip mutation on a binary-encoded individual.

        Parameters:
        - individual (list of int): The binary-encoded individual to mutate.
        - mutation_rate (float): The probability of flipping each bit.

        Returns:
        - list of int: The mutated individual.
        """
        mutated_individual = individual[:]
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i]  # Flip the bit
        return mutated_individual

    @staticmethod
    def swap_mutation(individual):
        """
        Performs swap mutation on an individual.

        Parameters:
        - individual (list): The individual to mutate.

        Returns:
        - list: The mutated individual with two randomly selected genes swapped.
        """
        mutated_individual = individual[:]
        idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
        return mutated_individual

    @staticmethod
    def inversion_mutation(individual):
        """
        Performs inversion mutation on an individual.

        Parameters:
        - individual (list): The individual to mutate.

        Returns:
        - list: The mutated individual with a randomly selected segment inverted.
        """
        mutated_individual = individual[:]
        start, end = sorted(random.sample(range(len(mutated_individual)), 2))
        mutated_individual[start:end+1] = reversed(mutated_individual[start:end+1])
        return mutated_individual

    @staticmethod
    def gaussian_mutation(individual, mutation_rate, mean=0, std_dev=1):
        """
        Performs Gaussian mutation on a real-valued individual.

        Parameters:
        - individual (list of float): The real-valued individual to mutate.
        - mutation_rate (float): The probability of mutating each gene.
        - mean (float): The mean of the Gaussian distribution (default is 0).
        - std_dev (float): The standard deviation of the Gaussian distribution (default is 1).

        Returns:
        - list of float: The mutated individual.
        """
        mutated_individual = individual[:]
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                mutated_individual[i] += random.gauss(mean, std_dev)
        return mutated_individual


# Importing the random module for mutation operations