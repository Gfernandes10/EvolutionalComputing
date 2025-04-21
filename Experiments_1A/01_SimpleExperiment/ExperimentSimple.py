# Add the root directory to the system path
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from MainOptimizationScript import MainOptimizationScript
results_dir = os.path.dirname(__file__)

fitness_function = 'Levi'
OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=fitness_function)
OptimizationObject.RESULTS_BASE_DIR = results_dir
OptimizationObject.multiple_optimization(num_executions=100, optimal_solution=[1, 1])


fitness_function = 'Drop-Wave'
OptimizationObjectDW = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=fitness_function)
OptimizationObjectDW.RESULTS_BASE_DIR = results_dir
OptimizationObjectDW.GENERATION_COUNT = 200
OptimizationObjectDW.multiple_optimization(num_executions=100, optimal_solution=[0, 0])