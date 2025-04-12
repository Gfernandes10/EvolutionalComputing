from MainOptimizationScript import MainOptimizationScript

OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION='Akley')
# OptimizationObject.initialize_optimization()
OptimizationObject.evaluate_performance(num_executions=100,optimal_solution=[0,0])