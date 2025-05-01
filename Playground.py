from MainOptimizationScript import MainOptimizationScript
size=200
num_executions=10
fitness_function='Levi'
identifier_prefix='LeviExperiment'
optimal_solution=[1, 1]

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=identifier_prefix)
OptimizationObject.ES_LAMBDA = 100
OptimizationObject.ES_MU = 20
OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_comma_lambda'
OptimizationObject.IDENTIFIER = identifier_prefix + 'mi_comma_lambda'
OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=identifier_prefix)
OptimizationObject.ES_LAMBDA = 100
OptimizationObject.ES_MU = 20
OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_plus_lambda'
OptimizationObject.IDENTIFIER = identifier_prefix + '_mi_plus_lambda'
OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

