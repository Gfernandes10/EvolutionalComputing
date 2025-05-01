from MainOptimizationScript import MainOptimizationScript

num_executions=10
fitness_function='Drop-Wave'
identifier_prefix='Drop-WaveExperiment'
optimal_solution=[0, 0]

# OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=identifier_prefix)
# OptimizationObject.ES_LAMBDA = 100
# OptimizationObject.ES_MU = 20
# OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_comma_lambda'
# OptimizationObject.IDENTIFIER = identifier_prefix + 'mi_comma_lambda'
# OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=identifier_prefix)
OptimizationObject.GENERATION_COUNT = 500
OptimizationObject.CHROMOSOME_LENGTH = 2
OptimizationObject.LOWER_BOUND = -5.0
OptimizationObject.UPPER_BOUND = 5.0
OptimizationObject.ES_LAMBDA = 100
OptimizationObject.ES_MU = 20
OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_plus_lambda'
OptimizationObject.IDENTIFIER = identifier_prefix + '_mi_plus_lambda'
OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

