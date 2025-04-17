from MainOptimizationScript import MainOptimizationScript

def experiment_population_sizes(sizes, num_executions, fitness_function, identifier_prefix):
    """
    Run experiments with different population sizes and collect results.
    :param sizes: List of population sizes to test.
    :param num_executions: Number of executions for each population size.
    :param fitness_function: Fitness function to use.
    :param identifier_prefix: Prefix for result folders.
    """
    for size in sizes:
        print(f"Running experiment with POPULATION_SIZE={size}")
        OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=f"{identifier_prefix}_POP{size}")
        OptimizationObject.POPULATION_SIZE = size
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=[1, 1])

# Define the population sizes to test
population_sizes = [50, 100, 200, 500]

# Run the experiments
experiment_population_sizes(
    sizes=population_sizes,
    num_executions=10,
    fitness_function='Levi',
    identifier_prefix='LeviExperiment'
)