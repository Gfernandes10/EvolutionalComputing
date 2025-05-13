import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from MainOptimizationScript import MainOptimizationScript
from Library.SaveResultsMethods import Results


# Drop-Wave function optimization experiment
fitness_function='Drop-Wave'
optimal_solution=[0, 0]
results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
os.makedirs(results_dir, exist_ok=True)


## Evolutionary Strategy (ES) with mi_plus_lambda and mibrho_plus_lambda
num_execution = 100
RESULTS_DW_MPL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
all_best_fitness_generation = []
all_step_size_generation = []
all_step_size_std = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = ['mi_plus_lambda', 'mibrho_plus_lambda']
for optimization_method in OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY:
    print(f"Running experiment with number of generations={optimization_method}")
    IDENTIFIER = optimization_method 
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 300
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.RECOMBINATION_FACTOR_RHO = 10
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = optimization_method
    OptimizationObject.RESULTS_BASE_DIR = results_dir
    # Run the optimization experiment
    OptimizationObject.multiple_optimization(num_executions=num_execution, optimal_solution=optimal_solution)
    RESULTS_DW_MPL.append(OptimizationObject.RESULTS)
    performance_metrics = RESULTS_DW_MPL[-1].PerformanceMetrics
    all_success_rates.append(performance_metrics['Success Rate (%)'])
    all_best_fitness_avg.append(performance_metrics['Average Best Fitness'])
    all_best_fitness_std.append(performance_metrics['Standard Deviation of Best Fitness'])
    all_best_mean_of_optimal_points.append(performance_metrics['Mean of Optimal Points'])
    all_best_std_of_optimal_points.append(performance_metrics['Standard Deviation of Optimal Points'])


for results in RESULTS_DW_MPL:
        for curves in results.Curves:
            if curves['Name'] == 'Aggregated Best Fitness Per Generation':
                all_best_fitness_generation.append(curves['Avg'])
                all_best_fitness_std.append(curves['Std'])
            if curves['Name'] == 'Aggregated Step Size Per Generation':
                all_step_size_generation.append(curves['Avg'])
                all_step_size_std.append(curves['Std'])

ResultsObject = Results()
ResultsObject.add_curve(
    x_data=range(len(all_best_fitness_generation[0])),
    x_label='Number of Generations', 
    y_data=all_best_fitness_generation,
    y_label='Best Fitness', 
    name='Best Fitness - (mi + lambda)',     
    plotType='line',
    y_std_data=all_best_fitness_std,
    CurveName= ['(μ+λ)', '(μ/ρ+λ)'],
)

ResultsObject.add_curve(
    x_data=range(len(all_step_size_generation[0])),
    x_label='Number of Generations', 
    y_data=all_step_size_generation,
    y_label='Step Size', 
    name ='Step Size - (mi + lambda)',
    plotType='line',
    y_std_data=all_step_size_std,
    CurveName= ['(μ+λ)', '(μ/ρ+λ)'],
)


ResultsObject.save_results(
    path=results_dir,
    overwrite=True
)


# Levi function optimization experiment
fitness_function='Levi'
optimal_solution=[1, 1]
results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
os.makedirs(results_dir, exist_ok=True)


## Evolutionary Strategy (ES) with mi_plus_lambda and mibrho_plus_lambda
num_execution = 100
RESULTS_DW_MPL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
all_best_fitness_generation = []
all_step_size_generation = []
all_step_size_std = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = ['mi_plus_lambda', 'mibrho_plus_lambda']
for optimization_method in OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY:
    print(f"Running experiment with number of generations={optimization_method}")
    IDENTIFIER = optimization_method 
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 300
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.RECOMBINATION_FACTOR_RHO = 10
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = optimization_method
    OptimizationObject.RESULTS_BASE_DIR = results_dir
    # Run the optimization experiment
    OptimizationObject.multiple_optimization(num_executions=num_execution, optimal_solution=optimal_solution)
    RESULTS_DW_MPL.append(OptimizationObject.RESULTS)
    performance_metrics = RESULTS_DW_MPL[-1].PerformanceMetrics
    all_success_rates.append(performance_metrics['Success Rate (%)'])
    all_best_fitness_avg.append(performance_metrics['Average Best Fitness'])
    all_best_fitness_std.append(performance_metrics['Standard Deviation of Best Fitness'])
    all_best_mean_of_optimal_points.append(performance_metrics['Mean of Optimal Points'])
    all_best_std_of_optimal_points.append(performance_metrics['Standard Deviation of Optimal Points'])


for results in RESULTS_DW_MPL:
        for curves in results.Curves:
            if curves['Name'] == 'Aggregated Best Fitness Per Generation':
                all_best_fitness_generation.append(curves['Avg'])
                all_best_fitness_std.append(curves['Std'])
            if curves['Name'] == 'Aggregated Step Size Per Generation':
                all_step_size_generation.append(curves['Avg'])
                all_step_size_std.append(curves['Std'])

ResultsObject = Results()
ResultsObject.add_curve(
    x_data=range(len(all_best_fitness_generation[0])),
    x_label='Number of Generations', 
    y_data=all_best_fitness_generation,
    y_label='Best Fitness', 
    name='Best Fitness - (mi + lambda)',     
    plotType='line',
    y_std_data=all_best_fitness_std,
    CurveName= ['(μ+λ)', '(μ/ρ+λ)'],
)

ResultsObject.add_curve(
    x_data=range(len(all_step_size_generation[0])),
    x_label='Number of Generations', 
    y_data=all_step_size_generation,
    y_label='Step Size', 
    name ='Step Size - (mi + lambda)',
    plotType='line',
    y_std_data=all_step_size_std,
    CurveName= ['(μ+λ)', '(μ/ρ+λ)'],
)

ResultsObject.save_results(
    path=results_dir,
    overwrite=True
)