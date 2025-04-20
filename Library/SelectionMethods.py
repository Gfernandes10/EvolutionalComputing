import random
import os
import sys
sys.dont_write_bytecode = True

class SelectionMethods:
    """
    This class is responsible for implementing various selection methods
    used to choose parents in genetic algorithms. These methods are
    essential for guiding the evolutionary process by selecting individuals
    based on their fitness or other criteria.
    """
    @staticmethod
    def tournament_selection(population_fitness, tournament_size=10, selected_parents=[]):
        """
        Implements tournament selection.
        A subset of individuals is chosen randomly, and the best among them is selected.

        :param population_fitness: List of tuples where each tuple contains an individual and its fitness value.
        :param tournament_size: Number of individuals to participate in each tournament (default is 10).
        :param selected_parents: List to store the selected parents (default is an empty list).
        :return: List of selected individuals (parents) after performing tournament selection.
        """

        for _ in range(len(population_fitness)):  # Loop through the population size
            tournament = random.sample(population_fitness, tournament_size-1)  # Randomly select individuals for the tournament
            winner = min(tournament, key=lambda x: x[1])  # Select the individual with the best fitness
            selected_parents.append(winner[0])  # Add the winner's chromosome to the selected parents list
        best_individual = min(population_fitness, key=lambda x: x[1])[0]  # Find the best individual in the population
        selected_parents.append(best_individual)  # Ensure the best individual is included in the selected parents  
        return selected_parents

    @staticmethod
    def roulette_wheel_selection(population_fitness):
        """
        Implements roulette wheel selection.
        Individuals are selected with probability proportional to their fitness.

        :param population_fitness: List of tuples (individual, fitness)
        :return: List of selected individuals (parents)
        """
        selected_parents = []
        total_fitness = sum(fitness for _, fitness in population_fitness)
        for _ in range(len(population_fitness)):
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual, fitness in population_fitness:
                current += fitness
                if current >= pick:
                    selected_parents.append(individual)
                    break
        return selected_parents
