import numpy as np


class EvolutionaryStrategies:
    @staticmethod
    def mi_c_lambda(population_fitness, selected_indices, n_children, step_size, best_so_far, best_solution):
        """
        Generate children by adding Gaussian noise to selected parents.

        Parameters:
        - population_fitness: List of tuples containing chromosomes and their fitness values.
        - selected_indices: Indices of the selected parents.
        - n_children: Number of children to generate per parent.
        - step_size: Step size for Gaussian noise.

        Returns:
        - children: List of generated children.
        - best_solution: Best solution found so far.
        - best_so_far: Best fitness value found so far.
        """
        children = []

        for i in selected_indices:
            # Check if this parent is the best solution ever seen
            if population_fitness[i][1] < best_so_far:
                best_so_far = population_fitness[i][1]
                best_solution = population_fitness[i][0]
                print(f"Best: f({best_solution}) = {best_so_far:.5f}")

            # Create children from the selected parents
            for _ in range(n_children):
                child = population_fitness[i][0] + np.random.randn(len(population_fitness[i][0])) * step_size
                children.append(child)

        return children, best_solution, best_so_far
    
    @staticmethod
    def mi_plus_lambda(population_fitness, selected_indices, n_children, step_size):
        """
        Generate children by adding Gaussian noise to selected parents and combine with the original population.

        Parameters:
        - population_fitness: List of tuples containing chromosomes and their fitness values.
        - selected_indices: Indices of the selected parents.
        - n_children: Number of children to generate per parent.
        - step_size: Step size for Gaussian noise.

        Returns:
        - combined_population: Combined population of parents and children.
        - best_solution: Best solution found so far.
        - best_so_far: Best fitness value found so far.
        """
        children = []
        best_so_far = float('inf')
        best_solution = None

        for i in selected_indices:
            # Check if this parent is the best solution ever seen
            if population_fitness[i][1] < best_so_far:
                best_so_far = population_fitness[i][1]
                best_solution = population_fitness[i][0]
                print(f"Best: f({best_solution}) = {best_so_far:.5f}")

            # Create children from the selected parents
            for _ in range(n_children):
                child = population_fitness[i][0] + np.random.randn(len(population_fitness[i][0])) * step_size
                children.append(child)

        # Combine parents and children
        combined_population = population_fitness + [(child, None) for child in children]

        return combined_population, best_solution

