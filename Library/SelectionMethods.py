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
    def inverted_roulette_wheel_selection(population_fitness):
        """
        Implements inverted roulette wheel selection.
        Individuals with lower fitness have a higher probability of being selected.

        :param population_fitness: List of tuples (individual, fitness)
        :return: List of selected individuals (parents)
        """
        selected_parents = []
        max_fitness = max(fitness for _, fitness in population_fitness)
        adjusted_fitness = [(individual, max_fitness - fitness) for individual, fitness in population_fitness]
        total_adjusted_fitness = sum(fitness for _, fitness in adjusted_fitness)

        for _ in range(len(population_fitness)):
            pick = random.uniform(0, total_adjusted_fitness)
            current = 0
            for individual, fitness in adjusted_fitness:
                current += fitness
                if current >= pick:
                    selected_parents.append(individual)
                    break
        return selected_parents

    @staticmethod
    def random_selection(population_fitness):
        """
        Implements random selection.
        Individuals are selected randomly without considering fitness.

        :param population_fitness: List of tuples (individual, fitness)
        :return: List of selected individuals (parents)
        """
        selected_parents = []
        for _ in range(len(population_fitness)):
            selected_parents.append(random.choice(population_fitness)[0])
        return selected_parents

    @staticmethod
    def deterministic_sampling_selection(population_fitness):
        """
        Implements deterministic sampling selection for minimization problems.
        Selects individuals based on a fixed proportion of fitness, ensuring proportional representation.
        Individuals with lower fitness have higher representation.

        :param population_fitness: List of tuples (individual, fitness)
        :return: List of selected individuals (parents)
        """
        selected_parents = []
        total_inverse_fitness = sum(1 / fitness for _, fitness in population_fitness)
        proportions = [(individual, (1 / fitness) / total_inverse_fitness) for individual, fitness in population_fitness]
        population_size = len(population_fitness)

        for individual, proportion in proportions:
            count = round(proportion * population_size)
            selected_parents.extend([individual] * count)

        # Adjust the number of selected parents to match the population size
        while len(selected_parents) > population_size:
            selected_parents.pop()
        while len(selected_parents) < population_size:
            selected_parents.append(random.choice(population_fitness)[0])

        return selected_parents
