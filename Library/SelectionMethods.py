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
    def roulette_wheel_selection(population, fitnesses):
        """
        Implements roulette wheel selection.
        Individuals are selected with a probability proportional to their fitness.

        :param population: List of individuals in the population.
        :param fitnesses: List of fitness values corresponding to the population.
        :return: Selected individual.
        """
        total_fitness = sum(fitnesses)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, fitness in zip(population, fitnesses):
            current += fitness
            if current > pick:
                return individual


    @staticmethod
    def rank_selection(population, fitnesses):
        """
        Implements rank-based selection.
        Individuals are ranked based on fitness, and selection probability
        is assigned based on rank.

        :param population: List of individuals in the population.
        :param fitnesses: List of fitness values corresponding to the population.
        :return: Selected individual.
        """
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
        ranks = range(1, len(sorted_population) + 1)
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        pick = random.random()
        for individual, cumulative_probability in zip(sorted_population, cumulative_probabilities):
            if pick <= cumulative_probability:
                return individual[0]