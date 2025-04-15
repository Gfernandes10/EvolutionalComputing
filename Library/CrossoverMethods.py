import random
import random
import random

class CrossoverMethods:
    """
    A class containing static methods to perform crossover operations 
    between two parents in a genetic algorithm.
    """

    @staticmethod
    def single_point_crossover(parent1, parent2):
        """
        Perform single-point crossover  between two parents.
        
        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.
        
        Returns:
            tuple: Two offspring resulting from the crossover.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length.")
        
        # Choose a random crossover point
        point = random.randint(1, len(parent1) - 1)
        
        # Create offspring by swapping segments
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        
        return offspring1, offspring2

    @staticmethod
    def uniform_crossover(parent1, parent2):
        """
        Perform uniform crossover between two parents.
        
        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.
        
        Returns:
            tuple: Two offspring resulting from the crossover.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length.")
        
        offspring1 = []
        offspring2 = []
        
        # Swap genes with 50% probability
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                offspring1.append(gene1)
                offspring2.append(gene2)
            else:
                offspring1.append(gene2)
                offspring2.append(gene1)
        
        return offspring1, offspring2

    @staticmethod
    def two_point_crossover(parent1, parent2):
        """
        Perform two-point crossover between two parents.
        
        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.
        
        Returns:
            tuple: Two offspring resulting from the crossover.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length.")
        
        # Choose two random crossover points
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        # Create offspring by swapping segments
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return offspring1, offspring2