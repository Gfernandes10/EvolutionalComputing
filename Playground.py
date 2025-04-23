from MainOptimizationScript import MainOptimizationScript
size=200
num_executions=10
fitness_function='Levi'
identifier_prefix='LeviExperiment'
optimal_solution=[1, 1]

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=identifier_prefix)
OptimizationObject.POPULATION_SIZE = size

OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)
