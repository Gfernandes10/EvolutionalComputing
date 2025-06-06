import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from MainOptimizationScript import MainOptimizationScript
from Library.SaveResultsMethods import Results


##################### Varying the number of executions #####################
num_executions= [10, 50, 100, 150, 200]


# Drop-Wave function optimization experiment
fitness_function='Drop-Wave'
optimal_solution=[0, 0]
results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
os.makedirs(results_dir, exist_ok=True)

## Evolutionary Strategy (ES) with mi_comma_lambda
RESULTS_DW_MCL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_comma_lambda'
for num_execution in num_executions:
    print(f"Running experiment with number of executions={num_execution}")
    IDENTIFIER = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY + "_NEXC" + str(num_execution)
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 500
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY
    OptimizationObject.RESULTS_BASE_DIR = results_dir
    OptimizationObject.multiple_optimization(num_executions=num_execution, optimal_solution=optimal_solution)
    RESULTS_DW_MCL.append(OptimizationObject.RESULTS)
    performance_metrics = RESULTS_DW_MCL[-1].PerformanceMetrics
    all_success_rates.append(performance_metrics['Success Rate (%)'])
    all_best_fitness_avg.append(performance_metrics['Average Best Fitness'])
    all_best_fitness_std.append(performance_metrics['Standard Deviation of Best Fitness'])
    all_best_mean_of_optimal_points.append(performance_metrics['Mean of Optimal Points'])
    all_best_std_of_optimal_points.append(performance_metrics['Standard Deviation of Optimal Points'])
    
ResultsObject = Results()
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_success_rates,
    y_label='Success Rate (%)', 
    name='Execs - Success Rate - (mi,lambda)',     
    plotType='bar',
)
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_best_fitness_avg,
    y_std_data=all_best_fitness_std,
    y_label='Average Best Fitness', 
    name ='Execs - Average Best Fitness - (mi,lambda)',
    plotType='errorbar',
)

mean_columns = list(zip(*all_best_mean_of_optimal_points))
std_columns = list(zip(*all_best_std_of_optimal_points))

for i, (mean_col, std_col) in enumerate(zip(mean_columns, std_columns)):
    ResultsObject.add_curve(
        x_data=num_executions,
        x_label='Number of Executions', 
        y_data=mean_col,
        y_std_data=std_col,
        y_label='Mean of Optimal Points', 
        name=f'Mean of Optimal Points - Gene {i} - (mi,lambda)',
        plotType='errorbar',
    )

ResultsObject.save_results(
    path=results_dir,
    overwrite=True
)


## Evolutionary Strategy (ES) with mi_plus_lambda
RESULTS_DW_MPL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_plus_lambda'
for num_execution in num_executions:
    print(f"Running experiment with number of executions={num_execution}")
    IDENTIFIER = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY + "_NEXC" + str(num_execution)
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 500
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY
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
    
ResultsObject = Results()
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_success_rates,
    y_label='Success Rate (%)', 
    name='Execs - Success Rate - (mi + lambda)',     
    plotType='bar',
)
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_best_fitness_avg,
    y_std_data=all_best_fitness_std,
    y_label='Average Best Fitness', 
    name ='Execs - Average Best Fitness - (mi + lambda)',
    plotType='errorbar',
)
mean_columns = list(zip(*all_best_mean_of_optimal_points))
std_columns = list(zip(*all_best_std_of_optimal_points))

for i, (mean_col, std_col) in enumerate(zip(mean_columns, std_columns)):
    ResultsObject.add_curve(
        x_data=num_executions,
        x_label='Number of Executions', 
        y_data=mean_col,
        y_std_data=std_col,
        y_label='Mean of Optimal Points', 
        name=f'Mean of Optimal Points - Gene {i} - (mi + lambda)',
        plotType='errorbar',
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

## Evolutionary Strategy (ES) with mi_comma_lambda
RESULTS_DW_MCL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_comma_lambda'
for num_execution in num_executions:
    print(f"Running experiment with number of executions={num_execution}")
    IDENTIFIER = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY + "_NEXC" + str(num_execution)
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 500
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY
    OptimizationObject.RESULTS_BASE_DIR = results_dir
    # Run the optimization experiment
    OptimizationObject.multiple_optimization(num_executions=num_execution, optimal_solution=optimal_solution)
    RESULTS_DW_MCL.append(OptimizationObject.RESULTS)
    performance_metrics = RESULTS_DW_MCL[-1].PerformanceMetrics
    all_success_rates.append(performance_metrics['Success Rate (%)'])
    all_best_fitness_avg.append(performance_metrics['Average Best Fitness'])
    all_best_fitness_std.append(performance_metrics['Standard Deviation of Best Fitness'])
    all_best_mean_of_optimal_points.append(performance_metrics['Mean of Optimal Points'])
    all_best_std_of_optimal_points.append(performance_metrics['Standard Deviation of Optimal Points'])
    
ResultsObject = Results()
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_success_rates,
    y_label='Success Rate (%)', 
    name='Execs - Success Rate - (mi,lambda)',     
    plotType='bar',
)
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_best_fitness_avg,
    y_std_data=all_best_fitness_std,
    y_label='Average Best Fitness', 
    name ='Execs - Average Best Fitness - (mi,lambda)',
    plotType='errorbar',
)

mean_columns = list(zip(*all_best_mean_of_optimal_points))
std_columns = list(zip(*all_best_std_of_optimal_points))

for i, (mean_col, std_col) in enumerate(zip(mean_columns, std_columns)):
    ResultsObject.add_curve(
        x_data=num_executions,
        x_label='Number of Executions', 
        y_data=mean_col,
        y_std_data=std_col,
        y_label='Mean of Optimal Points', 
        name=f'Mean of Optimal Points - Gene {i} - (mi,lambda)',
        plotType='errorbar',
    )

ResultsObject.save_results(
    path=results_dir,
    overwrite=True
)


## Evolutionary Strategy (ES) with mi_plus_lambda
RESULTS_DW_MPL = []
all_success_rates = []
all_best_fitness_avg = []
all_best_fitness_std = []
all_best_mean_of_optimal_points = []
all_best_std_of_optimal_points = []
OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_plus_lambda'
for num_execution in num_executions:
    print(f"Running experiment with number of executions={num_execution}")
    IDENTIFIER = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY + "_NEXC" + str(num_execution)
    OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=IDENTIFIER)
    OptimizationObject.OPTIMIZATION_METHOD = 'EvolutionaryStrategy'
    OptimizationObject.GENERATION_COUNT = 500
    OptimizationObject.CHROMOSOME_LENGTH = 2
    OptimizationObject.LOWER_BOUND = -5.0
    OptimizationObject.UPPER_BOUND = 5.0
    OptimizationObject.ES_LAMBDA = 100
    OptimizationObject.ES_MU = 20
    OptimizationObject.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY
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
    
ResultsObject = Results()
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_success_rates,
    y_label='Success Rate (%)', 
    name='Execs - Success Rate - (mi + lambda)',     
    plotType='bar',
)
ResultsObject.add_curve(
    x_data=num_executions,
    x_label='Number of Executions', 
    y_data=all_best_fitness_avg,
    y_std_data=all_best_fitness_std,
    y_label='Average Best Fitness', 
    name ='Execs - Average Best Fitness - (mi + lambda)',
    plotType='errorbar',
)
mean_columns = list(zip(*all_best_mean_of_optimal_points))
std_columns = list(zip(*all_best_std_of_optimal_points))

for i, (mean_col, std_col) in enumerate(zip(mean_columns, std_columns)):
    ResultsObject.add_curve(
        x_data=num_executions,
        x_label='Number of Executions', 
        y_data=mean_col,
        y_std_data=std_col,
        y_label='Mean of Optimal Points', 
        name=f'Mean of Optimal Points - Gene {i} - (mi + lambda)',
        plotType='errorbar',
    )

ResultsObject.save_results(
    path=results_dir,
    overwrite=True
)
