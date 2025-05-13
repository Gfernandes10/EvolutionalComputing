import numpy as np


class EvolutionaryStrategies:
    @staticmethod
    def mi_comma_lambda(MainOptObj, population_fitness, selected_indices, 
                       n_children, step_size, success_history, 
                       success_window, c = 0.85):
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
        success = 0
        LOWER_BOUND = MainOptObj.LOWER_BOUND
        UPPER_BOUND = MainOptObj.UPPER_BOUND

        for i in selected_indices:
            parent = population_fitness[i][0]
            parent_fitness = population_fitness[i][1]

            for _ in range(n_children):
                child = None
                while child is None or not EvolutionaryStrategies.in_bounds(child, LOWER_BOUND, UPPER_BOUND):
                    mutation = np.random.randn(len(parent)) * step_size
                    child = parent + mutation

                # Check if the child is better than the parent
                child_fitness = MainOptObj.evaluate_fitness(child)
                if child_fitness < parent_fitness:
                    success = 1
                else:
                    success = 0

                child_population_fitness = (child, child_fitness)        
                children.append(child_population_fitness)

        #Save success history        
        success_history.append(success)
        
        # Apply the 1/5 success rule
        step_size, success_history = EvolutionaryStrategies.apply_one_fifth_rule(
            success_history, step_size, c, success_window
        )

        return children, success_history, step_size
    
    @staticmethod
    def mi_plus_lambda(MainOptObj, population_fitness, selected_indices, 
                       n_children, step_size, success_history, 
                       success_window, c = 0.85):
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
        success = 0
        LOWER_BOUND = MainOptObj.LOWER_BOUND
        UPPER_BOUND = MainOptObj.UPPER_BOUND

        for i in selected_indices:
            parent = population_fitness[i][0]
            parent_fitness = population_fitness[i][1]
            # Keep parent in the population
            children.append(population_fitness[i])

            # Create children from the selected parents
            for _ in range(n_children):
                child = None
                while child is None or not EvolutionaryStrategies.in_bounds(child, LOWER_BOUND, UPPER_BOUND):
                    mutation = np.random.randn(len(parent)) * step_size
                    child = parent + mutation

                # Check if the child is better than the parent
                child_fitness = MainOptObj.evaluate_fitness(child)
                if child_fitness < parent_fitness:
                    success = 1
                else:
                    success = 0

                child_population_fitness = (child, child_fitness)
                children.append(child_population_fitness)

        #Save success history        
        success_history.append(success)
        
        # Apply the 1/5 success rule
        step_size, success_history = EvolutionaryStrategies.apply_one_fifth_rule(
            success_history, step_size, c, success_window
        )

        return children, success_history, step_size
    
    @staticmethod
    def mibrho_plus_lambda(MainOptObj, population_fitness, selected_indices, 
                           n_children, step_size, success_history, 
                           success_window, c=0.85, rho=6):
        """
        Generate children by recombining selected parents (intermediate recombination),
        adding Gaussian noise, and combining with the original population.

        Parameters:
        - population_fitness: List of tuples containing chromosomes and their fitness values.
        - selected_indices: Indices of the selected parents.
        - n_children: Number of children to generate per parent.
        - step_size: Step size for Gaussian noise.
        - rho: Number of parents to participate in recombination (default 2).

        Returns:
        - combined_population: Combined population of parents and children.
        - success_history: Updated success history.
        - step_size: Updated step size.
        """
        children = []
        success = 0
        LOWER_BOUND = MainOptObj.LOWER_BOUND
        UPPER_BOUND = MainOptObj.UPPER_BOUND

        # Randomly select `rho` parents for recombination
        if len(selected_indices) > rho:
            recombination_indices = np.random.choice(selected_indices, rho, replace=False)
        else:
            recombination_indices = selected_indices

        # Perform recombination and mutation
        for i in selected_indices:
            parent = population_fitness[i][0]
            parent_fitness = population_fitness[i][1]
            children.append(population_fitness[i])  # Keep parent in the population
            # If the parent is in the recombination group, perform intermediate recombination
            if i in recombination_indices:
                # Select other parents for recombination
                other_parents = [population_fitness[j][0] for j in recombination_indices if j != i]
                if other_parents:
                    # Perform intermediate recombination
                    child = np.mean([parent] + other_parents, axis=0)
                else:
                    child = parent.copy()  # No other parents available
            else:
                child = parent.copy()  # Parent not in recombination group

            # Apply mutation to the recombined or original parent
            for _ in range(n_children):
                mutated_child = None
                while mutated_child is None or not EvolutionaryStrategies.in_bounds(mutated_child, LOWER_BOUND, UPPER_BOUND):
                    mutation = np.random.randn(len(child)) * step_size
                    mutated_child = child + mutation

                # Evaluate the fitness of the mutated child
                child_fitness = MainOptObj.evaluate_fitness(mutated_child)
                if child_fitness < parent_fitness:
                    success = 1
                else:
                    success = 0

                children.append((mutated_child, child_fitness))

        # Save success history
        success_history.append(success)

        # Apply the 1/5 success rule
        step_size, success_history = EvolutionaryStrategies.apply_one_fifth_rule(
            success_history, step_size, c, success_window
        )


        return children, success_history, step_size

    @staticmethod
    def in_bounds(chromosome, LOWER_BOUND, UPPER_BOUND):
        """
        Check if a chromosome is within the defined bounds.
        """
        return all(LOWER_BOUND <= gene <= UPPER_BOUND for gene in chromosome)

    @staticmethod
    def apply_one_fifth_rule(success_history, step_size, c=0.85, success_window=10):
        """
        Apply the 1/5 success rule to adjust the mutation step size.

        Parameters:
        - success_history: List of success rates across generations.
        - step_size: Current mutation step size.
        - c: Adjustment factor for step size (default 0.85).
        - success_window: Window size for success rate averaging (default 10).

        Returns:
        - step_size: Updated step size.
        - success_history: Updated success history (reset if window is reached).
        """
        if len(success_history) >= success_window:
            avg_success = np.mean(success_history[-success_window:])
            if avg_success > 1 / 5:
                step_size /= c  # Increase step size
            elif avg_success < 1 / 5:
                step_size *= c  # Decrease step size
            success_history = []  # Reset success history
        return step_size, success_history

