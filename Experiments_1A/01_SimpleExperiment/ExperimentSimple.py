# Add the root directory to the system path
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from MainOptimizationScript import MainOptimizationScript

fitness_function = 'Levi'
results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=fitness_function)
OptimizationObject.RESULTS_BASE_DIR = results_dir
OptimizationObject.multiple_optimization(num_executions=100, optimal_solution=[1, 1])


fitness_function = 'Drop-Wave'
results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=fitness_function)
OptimizationObject.RESULTS_BASE_DIR = results_dir
OptimizationObject.multiple_optimization(num_executions=100, optimal_solution=[0, 0])