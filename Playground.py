from MainOptimizationScript import MainOptimizationScript

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION='Akley')
# OptimizationObject.initialize_optimization()
OptimizationObject.multiple_optimization(num_executions=10,optimal_solution=[0,0])