from Library.SelectionMethods import SelectionMethods
from Library.CrossoverMethods import CrossoverMethods
from Library.MutationMethods import MutationMethods
from Library.SaveResultsMethods import Results  # Import the Results class
from Library.EvolutionaryStrategies import EvolutionaryStrategies  # Import the EvolutionaryStrategies class
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
import time  # Import for execution time measurement
import os  # Import for file and directory handling
import json  # Import for saving configuration as JSON
from datetime import datetime  # Import for timestamp
from scipy.spatial.distance import pdist  # Import for optimized pairwise distances
from cma import CMA

class MainOptimizationScript:
    def __init__(self, FITNESS_FUNCTION_SELECTION, IDENTIFIER=None):
        """
        Constructor to initialize the class attributes.
        """
        # Class Parameters
        self.ENABLE_FITNESS_FUNCTION_VISUALIZATION = False
        self.ALLOWED_FITNESS_FUNCTIONS = ['Base', 'Akley', 'Drop-Wave','Levi']
        self.ResultsOverall = []  # Store performance data for all executions
        self.BestResult = None   # Store the best execution result
        self.diversity_per_generation = []  # Store diversity metrics
        self.best_fitness_per_generation = []  # Store best fitness per generation
        self.figures = []  # List to store figures for saving
        self.RESULTS = Results()  # Initialize the RESULTS property
        self.ENABLE_SAVE_RESULTS_AUTOMATICALLY = True  # Flag to enable automatic saving of results
        self.best_fitness_per_generation = []
        self.step_size_per_generation = []  # Store step size per generation
        self.diversity_per_generation = []

        # Validate FITNESS_FUNCTION_SELECTION
        if FITNESS_FUNCTION_SELECTION not in self.ALLOWED_FITNESS_FUNCTIONS:
            raise ValueError(f"Invalid FITNESS_FUNCTION_SELECTION. Allowed values are: {self.ALLOWED_FITNESS_FUNCTIONS}")

        # Initialize configuration parameters
        self.POPULATION_SIZE = 200
        self.GENERATION_COUNT = 500        
        self.CHROMOSOME_LENGTH = 2
        self.LOWER_BOUND = -5
        self.UPPER_BOUND = 5
        self.FITNESS_FUNCTION_SELECTION = FITNESS_FUNCTION_SELECTION
        self.SELECTION_METHOD = 'Random'
        self.SELECTION_TOURNAMENT_SIZE = 10
        self.CROSSOVER_METHOD = 'Random'
        self.CROSSOVER_RATE = 0.8
        self.MUTATION_METHOD = 'Random'
        self.MUTATION_RATE = 0.1
        self.APPLY_DIVERSITY_MAINTENANCE = True  # Flag to apply diversity maintenance strategies
        self.OPTIMIZATION_METHOD = 'EvolutionaryStrategy' #Options: 'GeneticAlgorithm_Elitism', 'EvolutionaryStrategy', 'CMAEStrategy'
        self.OPTIMIZATION_METHOD_NUMBER_ELITES = 20
        self.IDENTIFIER = IDENTIFIER  # Optional identifier for result folder prefix
        self.STOPPING_METHOD = 'GenerationCount'  # Options: 'GenerationCount', 'TargetFitness', 'NoImprovement'
        self.TARGET_FITNESS = None  # Desired fitness value for stopping
        self.NO_IMPROVEMENT_LIMIT = None  # Max generations without improvement

        ## Parameters for Evolutionary Strategy
        self.ES_MU = 20  # Number of parents in ES
        self.ES_LAMBDA = 100  # Number of offspring in ES
        self.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY = 'mi_comma_lambda'  # Options: 'mi_comma_lambda', 'mi_plus_lambda', 'mibrho_plus_lambda'
        self.RECOMBINATION_FACTOR_RHO = 10  # Default rho value for mibrho_plus_lambda

        ## Parameters for CMA-ES
        self.CMA_OBJ = None
        self.CMA_FITNESS_FUNCTION = None
        self.CMA_INITIAL_STEP_SIZE = None


    def evaluate_fitness(self,chromosome):
        match self.FITNESS_FUNCTION_SELECTION:
            case 'Base':
            # Base fitness function 
                x = chromosome
                f1= x[0] + 2 * (-x[1]) + 3
                f2= 2 * x[0] + x[1] - 8
                fitness_value = f1**2+f2**2
                ENABLE_FITNESS_FUNCTION_VISUALIZATION = True
            case 'Akley':
                x = chromosome[0]
                y = chromosome[1]
                fitness_value = -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20
                ENABLE_FITNESS_FUNCTION_VISUALIZATION = True
            case 'Drop-Wave':
                x = chromosome[0]
                y = chromosome[1]
                numerator = 1 + cos(12 * sqrt(x**2 + y**2))
                denominator = 0.5 * (x**2 + y**2) + 2
                fitness_value = - numerator / denominator
                ENABLE_FITNESS_FUNCTION_VISUALIZATION = True
            case 'Levi':
                x = chromosome[0]
                y = chromosome[1]
                term1 = np.sin(3 * np.pi * x)**2
                term2 = (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
                term3 = (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
                fitness_value = term1 + term2 + term3
                ENABLE_FITNESS_FUNCTION_VISUALIZATION = True
            case _:
                raise ValueError("Invalid FITNESS_FUNCTION_SELECTION")
        return fitness_value

    def multiple_optimization(self, num_executions, optimal_solution=None, tolerance=1e-2):
        """
        Evaluate the performance of the optimization algorithm by running it multiple times and analyzing the results.
        Parameters:
            num_executions (int): Number of times to run the optimization.
            optimal_solution (list, optional): Known optimal solution value (if available). Defaults to None.
            tolerance (float, optional): Tolerance for determining success in finding the optimal solution. Defaults to 1e-2.
        Attributes:
            ResultsOverall (list): Stores the results of all executions, including the best solution and fitness.
            BestResult (dict): Stores the best solution and fitness found across all executions.
            best_fitness_values (list): Stores the best fitness values for each execution.
            all_best_fitness_per_generation (list): Stores the best fitness per generation for all executions.
            all_diversity_per_generation (list): Stores diversity per generation for all executions.
        Performance Metrics:
            avg_best_fitness (float): Average of the best fitness values across all executions.
            success_rate (float): Percentage of executions that successfully found a solution within the tolerance of the optimal solution.
            execution_time (float): Total time taken to complete all executions.
            mean_optimal_point (numpy.ndarray): Mean of the optimal points found across all executions.
            std_optimal_point (numpy.ndarray): Standard deviation of the optimal points found across all executions.
        Visualization:
            - Plots the convergence curve with mean and standard deviation.
            - Plots population diversity with mean and standard deviation.
            - Visualizes the mean and standard deviation of optimal points.
        Saves:
            - Results of all executions.
            - Configuration parameters used for the optimization.
            - Performance metrics.
            - Aggregated metrics for convergence curves and diversity.
            - Optimal points and their standard deviations.
        Returns:
            None
        """
        success_count = 0
        best_fitness_values = []  # Store best fitness values for each execution
        optimal_points = []  # Store all optimal points found
        self.ResultsOverall = []  # Reset results for new executions
        best_overall_fitness = float('inf')  # Initialize best fitness as infinity
        all_best_fitness_per_generation = []  # Store best fitness per generation for all executions
        all_diversity_per_generation = []  # Store diversity per generation for all executions
        all_step_size_per_generation = []  # Store step size per generation for all executions

        cma_all_generations = []
        cma_all_step_sizes = []
        cma_all_gene = []

        self.visualize_fitness_function()

        start_time = time.time()  # Start timing        
        for execution in range(1, num_executions + 1):
            match self.OPTIMIZATION_METHOD:
                case 'GeneticAlgorithm_Elitism':
                    best_population_fitness = self.elitism_optimization()
                case 'EvolutionaryStrategy':
                    best_population_fitness = self.evolutionary_strategy_optimization()
                case 'CMAEStrategy':
                    best_population_fitness = self.cmaes_strategy_optimization()
                case _:
                    raise ValueError(f"Invalid OPTIMIZATION_METHOD: {self.OPTIMIZATION_METHOD}")

            # Retrieve the best solution and its fitness value from the last generation
            best_solution = best_population_fitness[0]
            best_fitness = best_population_fitness[1]

            # Store the result of this execution
            self.ResultsOverall.append({
                "BestSolution": best_solution,
                "BestFitness": best_fitness
            })

            # Update the best result if this execution is better
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                self.BestResult = {
                    "BestSolution": best_solution,
                    "BestFitness": best_fitness
                }
                if self.CMA_OBJ is not None:
                    CMA_TRACE = self.CMA_OBJ.trace
                    NUM_GENERATIONS = list(range(1, len(CMA_TRACE) + 1))
                    GENE = [entry['m'] for entry in CMA_TRACE]
                    STEP_SIZE = [entry['σ'] for entry in CMA_TRACE]
                    

            # Print progress information
            elapsed_time = time.time() - start_time
            print(f"Execution {execution}/{num_executions} completed. Best Fitness: {best_fitness:.6f}. Best Solution: {best_solution} Elapsed Time: {elapsed_time:.2f} seconds")

            best_fitness_values.append(best_fitness)



            distance = np.linalg.norm(np.array(best_solution) - np.array(optimal_solution)) 
            if distance <= tolerance:
                success_count += 1


            optimal_points.append(best_solution)
            all_best_fitness_per_generation.append(self.best_fitness_per_generation)
            all_diversity_per_generation.append(self.diversity_per_generation)
            all_step_size_per_generation.append(self.step_size_per_generation)


        end_time = time.time()  # End timing
        execution_time = end_time - start_time

        if all_best_fitness_per_generation and any(all_best_fitness_per_generation):
            self.RESULTS.add_curve(
                x_data=range(len(all_best_fitness_per_generation[0])),
                y_data=all_best_fitness_per_generation,
                x_label="Generation",
                y_label="Average Best Fitness",
                name="Aggregated Best Fitness Per Generation",
                plot_avg=True,
                plot_std=True
            )
        if all_diversity_per_generation and any(all_diversity_per_generation):
            self.RESULTS.add_curve(
                x_data=range(len(all_diversity_per_generation[0])),
                y_data=all_diversity_per_generation,
                x_label="Generation",
                y_label="Average Diversity",
                name="Aggregated Diversity Per Generation",
                plot_avg=True,
                plot_std=True
            )
            
        if self.OPTIMIZATION_METHOD == 'EvolutionaryStrategy':
            if all_step_size_per_generation and any(all_step_size_per_generation):
                self.RESULTS.add_curve(
                    x_data=range(len(all_step_size_per_generation[0])),
                    y_data=all_step_size_per_generation,
                    x_label="Generation",
                    y_label="Step Size",
                    name="Aggregated Step Size Per Generation",
                    plot_avg=True,
                    plot_std=True
                )
        
        if self.CMA_OBJ is not None:
            self.RESULTS.add_curve(
                x_data=NUM_GENERATIONS,
                y_data=STEP_SIZE,
                x_label="Generation",
                y_label="Step Size",
                name="CMA-ES Step Size Per Generation",
            )

            for gene_index in range(len(GENE[0])):
                # Extract the gene values for the current index across all generations
                gene = [gene[gene_index] for gene in GENE]
                self.RESULTS.add_curve(
                    x_data=NUM_GENERATIONS,
                    y_data=gene,
                    x_label="Generation",
                    y_label=f"Gene {gene_index + 1}",
                    name=f"CMA-ES Gene {gene_index + 1} Per Generation",
                )
        # Calculate mean and standard deviation of optimal points
        #ONLY FOR 2D PROBLEMS
        optimal_points = np.array(optimal_points)
        self.RESULTS.add_curve(
            x_data=optimal_points[:,0],
            y_data=optimal_points[:,1],
            x_label="X Coordinate",
            y_label="Y Coordinate",
            name="Optimal Points Distribution",
            plot_avg=True,
            plot_std=True,
            plotType="scatter"
        )

        mean_optimal_point = self.RESULTS.Curves[-1]["Avg"]
        std_optimal_point = self.RESULTS.Curves[-1]["Std"]

        # Calculate performance metrics
        avg_best_fitness = np.mean(best_fitness_values)
        std_best_fitness = np.std(best_fitness_values)
        success_rate = success_count / num_executions


        print(f"Performance Metrics:")
        print(f"Total Execution Time: {execution_time:.2f} seconds")        
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Average Best Fitness: {avg_best_fitness:.6f}")
        print(f"Standard Deviation of Best Fitness: {std_best_fitness:.6f}")
        print(f"Best Solution Found: {self.BestResult['BestFitness']}")
        print(f"Chromosome for Best Solution: {self.BestResult['BestSolution']}")
        print(f"Mean of Optimal Points: {mean_optimal_point}")
        print(f"Standard Deviation of Optimal Points: {std_optimal_point}")

        performance_metrics = {
            "Total Execution Time (s)": float(execution_time),
            "Success Rate (%)": float(success_rate * 100),
            "Average Best Fitness": float(avg_best_fitness),
            "Standard Deviation of Best Fitness": float(std_best_fitness),
            "Best Solution Found": float(self.BestResult['BestFitness']),
            "Chromosome for Best Solution": [float(x) for x in self.BestResult['BestSolution']],
            "Mean of Optimal Points": [float(x) for x in mean_optimal_point.tolist()],  # Convert numpy array to list of floats
            "Standard Deviation of Optimal Points": [float(x) for x in std_optimal_point.tolist()]  # Convert numpy array to list of floats
        }

        # Save results, configuration, and figures
        match self.OPTIMIZATION_METHOD:
            case 'GeneticAlgorithm_Elitism':
                config = {
                            "POPULATION_SIZE": self.POPULATION_SIZE,
                            "GENERATION_COUNT": self.GENERATION_COUNT,
                            "CHROMOSOME_LENGTH": self.CHROMOSOME_LENGTH,
                            "LOWER_BOUND": self.LOWER_BOUND,
                            "UPPER_BOUND": self.UPPER_BOUND,
                            "FITNESS_FUNCTION_SELECTION": self.FITNESS_FUNCTION_SELECTION,
                            "SELECTION_METHOD": self.SELECTION_METHOD,
                            "SELECTION_TOURNAMENT_SIZE": self.SELECTION_TOURNAMENT_SIZE,
                            "CROSSOVER_METHOD": self.CROSSOVER_METHOD,
                            "CROSSOVER_RATE": self.CROSSOVER_RATE,
                            "MUTATION_METHOD": self.MUTATION_METHOD,
                            "MUTATION_RATE": self.MUTATION_RATE,
                            "OPTIMIZATION_METHOD": self.OPTIMIZATION_METHOD,
                            "OPTIMIZATION_METHOD_NUMBER_ELITES": self.OPTIMIZATION_METHOD_NUMBER_ELITES,
                            "NUM_EXECUTIONS": num_executions,
                            "OPTIMAL_SOLUTION": optimal_solution,
                            "TOLERANCE": tolerance
                        }
            case 'EvolutionaryStrategy':
                config = {
                            "GENERATION_COUNT": self.GENERATION_COUNT,
                            "CHROMOSOME_LENGTH": self.CHROMOSOME_LENGTH,
                            "LOWER_BOUND": self.LOWER_BOUND,
                            "UPPER_BOUND": self.UPPER_BOUND,
                            "FITNESS_FUNCTION_SELECTION": self.FITNESS_FUNCTION_SELECTION,
                            "MUTATION_METHOD": self.MUTATION_METHOD,
                            "OPTIMIZATION_METHOD": self.OPTIMIZATION_METHOD,
                            "OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY": self.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY,
                            "ES_MU": self.ES_MU,
                            "ES_LAMBDA": self.ES_LAMBDA,
                            "RECOMBINATION_FACTOR_RHO": self.RECOMBINATION_FACTOR_RHO,
                            "NUM_EXECUTIONS": num_executions,
                            "OPTIMAL_SOLUTION": optimal_solution,
                            "TOLERANCE": tolerance
                        }
            case 'CMAEStrategy':
                config = {
                            "GENERATION_COUNT": self.GENERATION_COUNT,
                            "CHROMOSOME_LENGTH": self.CHROMOSOME_LENGTH,
                            "POPULATION_SIZE": self.POPULATION_SIZE,
                            "LOWER_BOUND": self.LOWER_BOUND,
                            "UPPER_BOUND": self.UPPER_BOUND,
                            "FITNESS_FUNCTION_SELECTION": self.FITNESS_FUNCTION_SELECTION,
                            "OPTIMIZATION_METHOD": self.OPTIMIZATION_METHOD,
                            "NUM_EXECUTIONS": num_executions,
                            "OPTIMAL_SOLUTION": optimal_solution,
                            "TOLERANCE": tolerance
                        }
            case _:
                raise ValueError(f"Invalid OPTIMIZATION_METHOD: {self.OPTIMIZATION_METHOD}")



        # Store performance metrics in RESULTS
        self.RESULTS.add_metric("Success Rate (%)", success_rate * 100)
        self.RESULTS.add_metric("Average Best Fitness", avg_best_fitness)
        self.RESULTS.add_metric("Best Solution Found", self.BestResult['BestFitness'])
        self.RESULTS.add_metric("Execution Time (s)", execution_time)


        # Store configuration in RESULTS
        self.RESULTS.set_config(config)
        self.RESULTS.set_performance(performance_metrics)

        # Save results
        folder_name = f"{self.IDENTIFIER}_" if self.IDENTIFIER else ""
        # Determine the base directory for saving results
        if hasattr(self, 'RESULTS_BASE_DIR') and self.RESULTS_BASE_DIR:
            base_dir = self.RESULTS_BASE_DIR
            results_dir = os.path.join(base_dir, f"{folder_name}")
        else:
            base_dir = "Results"  # Default directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_dir = os.path.join(base_dir, f"{folder_name}{timestamp}")
        
        if self.ENABLE_SAVE_RESULTS_AUTOMATICALLY:
            self.RESULTS.save_results(
                path=results_dir,
                overwrite=True
            )
    def cmaes_strategy_optimization(self):
        """
        Perform optimization using the CMA-ES strategy.
        """
        # Initialize CMA-ES parameters
        # Check if CMA-ES parameters are specified
        if self.CMA_FITNESS_FUNCTION is None or self.CMA_INITIAL_STEP_SIZE is None:
            raise ValueError("CMA-ES parameters must be specified: CMA_FITNESS_FUNCTION, CMA_INITIAL_STEP_SIZE, and CMA_INITIAL_SOLUTION.")
        initial_solution = self.generate_chromosome()
        initial_step_size = self.CMA_INITIAL_STEP_SIZE
        fitness_fn = self.CMA_FITNESS_FUNCTION
        max_epoch = self.GENERATION_COUNT

        self.CMA_OBJ = CMA(initial_solution=initial_solution,
                  initial_step_size=initial_step_size,
                    fitness_function=fitness_fn,
                    store_trace=True,
                    population_size=self.POPULATION_SIZE)

        best_solution, best_fitness = self.CMA_OBJ.search(max_epoch)

        
        if isinstance(best_solution, np.ndarray):
            best_solution = best_solution.tolist()
        best_population_fitness = (best_solution, best_fitness)
        return best_population_fitness 
    def evolutionary_strategy_optimization(self):
        """
        Perform optimization using the evolutionary strategy method.
        """
        mu = self.ES_MU
        lambda_ = self.ES_LAMBDA
        c = 0.85  # Default success probability for ES
        rho = self.RECOMBINATION_FACTOR_RHO # Default rho value for mibrho_plus_lambda
        success_history = []
        success_window = 10*self.CHROMOSOME_LENGTH  # Default success window size
        # step_size = (self.UPPER_BOUND - self.LOWER_BOUND) / np.sqrt(self.CHROMOSOME_LENGTH)  # Initial step size based on chromosome length
        step_size = (self.UPPER_BOUND - 0) / np.sqrt(self.CHROMOSOME_LENGTH)
        population = []
        self.best_fitness_per_generation = []
        self.step_size_per_generation = []  # Store step size per generation
        self.diversity_per_generation = []
        self.euclidean_diversity_per_generation = []  # Store Euclidean diversity metrics
        best_so_far = float('inf')
        best_solution = None
        n_children = int(lambda_/mu) #Number of children per parent 
        no_improvement_count = 0
       
        for _ in range(lambda_):
            candidate = None
            while candidate is None or not self.in_bounds(candidate):
                candidate = self.generate_chromosome()
            population.append(candidate)
        population_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in population]
        
        for epoch in range(self.GENERATION_COUNT):
            
            selected_indices = sorted(range(len(population_fitness)), key=lambda i: population_fitness[i][1])[:mu]
            match self.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY:
                case 'mi_comma_lambda':
                    population_fitness, success_history, step_size = EvolutionaryStrategies.mi_comma_lambda(self,
                                                                                    population_fitness, 
                                                                                    selected_indices, 
                                                                                    n_children, 
                                                                                    step_size,
                                                                                    success_history,
                                                                                    success_window,
                                                                                    c)
                case 'mi_plus_lambda':
                    population_fitness, success_history, step_size = EvolutionaryStrategies.mi_plus_lambda(self,
                                                                                    population_fitness, 
                                                                                    selected_indices, 
                                                                                    n_children, 
                                                                                    step_size,
                                                                                    success_history,
                                                                                    success_window,
                                                                                    c)
                case 'mibrho_plus_lambda':
                    population_fitness, success_history, step_size = EvolutionaryStrategies.mibrho_plus_lambda(self,
                                                                                    population_fitness, 
                                                                                    selected_indices, 
                                                                                    n_children, 
                                                                                    step_size,
                                                                                    success_history,
                                                                                    success_window,
                                                                                    c,
                                                                                    rho)
                case _:
                    raise ValueError(f"Invalid OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY: {self.OPTIMIZATION_METHOD_EVOLUTIONARY_STRATEGY}")
            
            current_best = min(population_fitness, key=lambda x: x[1])
            if current_best[1] < best_so_far:
                best_so_far = current_best[1]
                best_solution = current_best[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Calculate and store diversity (standard deviation)
            gene_matrix = np.array([chromosome for chromosome, _ in population_fitness])  # Extract genes into a matrix
            diversity = np.mean(np.std(gene_matrix, axis=0))  # Calculate mean of standard deviations across dimensions
            self.diversity_per_generation.append(diversity)

            # Store fitness values for the current generation
            self.best_fitness_per_generation.append(best_so_far)

            # Store step size for the current generation
            self.step_size_per_generation.append(step_size)

            # Stopping criteria
            if self.STOPPING_METHOD == 'TargetFitness' and self.TARGET_FITNESS is not None:
                if best_so_far <= self.TARGET_FITNESS:
                    print(f"Stopping early: Target fitness {self.TARGET_FITNESS} reached at generation {epoch + 1}.")
                    break
            elif self.STOPPING_METHOD == 'NoImprovement' and self.NO_IMPROVEMENT_LIMIT is not None:
                if no_improvement_count >= self.NO_IMPROVEMENT_LIMIT:
                    print(f"Stopping early: No improvement for {self.NO_IMPROVEMENT_LIMIT} generations.")
                    break
        
        # Ensure the best solution is returned as a list, not a numpy array
        if isinstance(best_solution, np.ndarray):
            best_solution = best_solution.tolist()
        best_population_fitness = (best_solution, best_so_far)
        return best_population_fitness            



    def elitism_optimization(self):
        population = []
        population = [self.generate_chromosome() for _ in range(self.POPULATION_SIZE)]
        population_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in population]
        self.best_fitness_per_generation = []
        self.diversity_per_generation = []  # Reset diversity metrics
        self.euclidean_diversity_per_generation = []  # Store Euclidean diversity metrics

        best_so_far = float('inf')
        no_improvement_count = 0

        for idx in range(self.GENERATION_COUNT):          
            #Select parents for crossover
            selected_parents = self.selection(population_fitness)
            #Create offspring through crossover and mutation
            offspring = []
            for i in range(0, self.POPULATION_SIZE, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutation(child1), self.mutation(child2)])
            new_population_fitness = [(chromosome, 0) for chromosome in offspring]
            
            #Apply elitism to retain the best individuals
            num_elites = self.OPTIMIZATION_METHOD_NUMBER_ELITES # Number of elite individuals to retain
            elites = sorted(population_fitness, key=lambda x: x[1])[:num_elites]

            # Update population
            population = [elite[0] for elite in elites] + [chromosome[0] for chromosome in new_population_fitness]

            #Evaluate fitness of the population
            population_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in population]
            self.population_fitness = population_fitness

            # Calculate and store diversity (standard deviation)
            gene_matrix = np.array([chromosome for chromosome, _ in population_fitness])  # Extract genes into a matrix
            diversity = np.mean(np.std(gene_matrix, axis=0))  # Calculate mean of standard deviations across dimensions
            self.diversity_per_generation.append(diversity)

            # Calculate and store diversity (Euclidean distance)
            if len(gene_matrix) > 1:  # Ensure there are at least two chromosomes
                euclidean_distances = pdist(gene_matrix, metric='euclidean')  # Efficient pairwise distances
                euclidean_diversity = np.mean(euclidean_distances)
            else:
                euclidean_diversity = 0  # No diversity if there's only one chromosome
            self.euclidean_diversity_per_generation.append(euclidean_diversity)

            # # Apply diversity maintenance strategies
            if self.APPLY_DIVERSITY_MAINTENANCE:
                # Maintain diversity in the population
                population_fitness = self.maintain_diversity(population_fitness, diversity)

            # Store best fitness for convergence curve
            best_population_fitness = min(population_fitness, key=lambda x: x[1])
            best_fitness = best_population_fitness[1]
            self.best_fitness_per_generation.append(best_fitness)

            # Check for improvement
            if best_fitness < best_so_far:
                best_so_far = best_fitness
                no_improvement_count = 0  # Reset counter
            else:
                no_improvement_count += 1

            # Stopping criteria
            if self.STOPPING_METHOD == 'TargetFitness' and self.TARGET_FITNESS is not None:
                if best_so_far <= self.TARGET_FITNESS:
                    print(f"Stopping early: Target fitness {self.TARGET_FITNESS} reached at generation {idx + 1}.")
                    break
            elif self.STOPPING_METHOD == 'NoImprovement' and self.NO_IMPROVEMENT_LIMIT is not None:
                if no_improvement_count >= self.NO_IMPROVEMENT_LIMIT:
                    print(f"Stopping early: No improvement for {self.NO_IMPROVEMENT_LIMIT} generations.")
                    break


        
        return best_population_fitness
    
    def maintain_diversity(self, population_fitness, diversity, threshold=0.1):
        """
        Apply strategies to maintain diversity in the population.

        Parameters:
        - population_fitness (list): List of tuples containing chromosomes and their fitness values.
        - diversity (float): Current diversity metric.
        - threshold (float): Minimum diversity threshold to trigger strategies.

        Returns:
        - list: Updated population_fitness with recalculated fitness values for new chromosomes.
        """
        if diversity < threshold:
            print(f"Diversity below threshold ({diversity:.4f} < {threshold}). Applying strategies...")

            # Extract population (chromosomes) from population_fitness
            population = [chromosome for chromosome, _ in population_fitness]

            # Strategy 1: Partial population reinicialization
            num_to_replace = int(0.2 * len(population))  # Replace 20% of the population
            new_individuals = [self.generate_chromosome() for _ in range(num_to_replace)]
            new_individuals_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in new_individuals]

            # Replace the worst individuals with new ones
            population_fitness = sorted(population_fitness, key=lambda x: x[1])[:-num_to_replace] + new_individuals_fitness

            # Strategy 2: Increase mutation rate temporarily
            self.MUTATION_RATE *= 1.5
            print(f"Mutation rate temporarily increased to {self.MUTATION_RATE:.2f}")

            # Strategy 3: Introduce random individuals
            num_random = int(0.1 * len(population))  # Add 10% random individuals
            random_individuals = [self.generate_chromosome() for _ in range(num_random)]
            random_individuals_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in random_individuals]

            # Add random individuals to the population
            population_fitness.extend(random_individuals_fitness)

        return population_fitness



    ########### Base functions ############

    def generate_chromosome(self):
        """
        Generate a random chromosome.
        """
        return [random.uniform(self.LOWER_BOUND, self.UPPER_BOUND) for _ in range(self.CHROMOSOME_LENGTH)]
    
    def selection(self, population_fitness):
        """
        Select parents for crossover using tournament selection.

        :param population_fitness: List of individuals containing chromosomes and their fitness values.
        """

        # Selection Strategies
        match self.SELECTION_METHOD:
            case 'TournamentSelection':
            # 1. Tournament Selection
                selected_parents = SelectionMethods.tournament_selection(population_fitness, tournament_size=self.SELECTION_TOURNAMENT_SIZE)
            case 'InvertRouletteWheelSelection':
            # 2. Roulette Wheel Selection
                selected_parents = SelectionMethods.inverted_roulette_wheel_selection(population_fitness)
            case 'RandomSelection':
            # 3. Random Selection
                selected_parents = SelectionMethods.random_selection(population_fitness)
            case 'DeterministicSamplingSelection':
            # 4. Deterministic Sampling Selection
                selected_parents = SelectionMethods.deterministic_sampling_selection(population_fitness)
            case 'Random':
            # 3. Random Selection
                self.SELECTION_METHOD = random.choice(['TournamentSelection', 
                                                       'InvertRouletteWheelSelection', 
                                                       'RandomSelection', 
                                                       'DeterministicSamplingSelection'])
                selected_parents = self.selection(population_fitness)
                self.SELECTION_METHOD = 'Random'  # Reset to default                        
            case _:
                selected_parents = []
                raise ValueError("Invalid SELECTION_METHOD")                
        return selected_parents  # Return the list of selected parents

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        """
        # Crossover Strategies
        if random.random() < self.CROSSOVER_RATE:
            match self.CROSSOVER_METHOD:
                case 'SinglePointCrossover':
                # 1. Single-point crossover
                    child1, child2 = CrossoverMethods.single_point_crossover(parent1, parent2)
                case 'ArithmeticCrossover':
                # 2. Arithmetic crossover
                    child1, child2 = CrossoverMethods.arithmetic_crossover(parent1, parent2)
                case 'Random':
                    match random.choice(['SinglePointCrossover', 'ArithmeticCrossover']):
                        case 'SinglePointCrossover':
                            self.CROSSOVER_METHOD = 'SinglePointCrossover'
                        case 'ArithmeticCrossover':
                            self.CROSSOVER_METHOD = 'ArithmeticCrossover'
                    child1, child2 = self.crossover(parent1, parent2)
                    self.CROSSOVER_METHOD = 'Random'  # Reset to default
                case _:
                    raise ValueError("Invalid CROSSOVER_METHOD")
            return child1, child2
        else:
            # No crossover, return parents as children
            return parent1, parent2
        
    def mutation(self, individual):
        """
        Perform mutation on a chromosome.
        """
        # Mutation Strategies
        match self.MUTATION_METHOD:
        # 1. Random mutation on individual genes
            case 'RandomMutationOnIndividualGenes':
                mutated_individual = MutationMethods.random_mutation_on_individual_genes(individual, self.MUTATION_RATE)
            case 'GaussianMutation':
                # 2. Gaussian mutation
                mutated_individual = MutationMethods.gaussian_mutation(individual, self.MUTATION_RATE)
            case 'Random':
                # 3. Random mutation method
                self.MUTATION_METHOD = random.choice(['RandomMutationOnIndividualGenes', 'GaussianMutation'])
                mutated_individual = self.mutation(individual)
                self.MUTATION_METHOD = 'Random'
            case _:
                raise ValueError("Invalid MUTATION_METHOD")
        return mutated_individual

    def visualize_fitness_function(self):
        """
        Visualize the fitness function in 3D.
        """
        if not self.ENABLE_FITNESS_FUNCTION_VISUALIZATION:
            print("Fitness function visualization is disabled.")
            return
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x = np.linspace(self.LOWER_BOUND, self.UPPER_BOUND, 100)
        y = np.linspace(self.LOWER_BOUND, self.UPPER_BOUND, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate_fitness([X, Y])
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title("Fitness Function Visualization")
        plt.show(block=False)  # Allow script to continue without blocking
    def in_bounds(self, chromosome):
        """
        Check if a chromosome is within the defined bounds.
        """
        return all(self.LOWER_BOUND <= gene <= self.UPPER_BOUND for gene in chromosome)
    def save_results(self, results, config, performance_metrics, curve_data, optimal_points, curve_std_data, optimal_points_std):
        """
        Save results, configuration, figures, performance metrics, curve data, and optimal points with their standard deviations to a timestamped folder.
        """
        folder_name = f"{self.IDENTIFIER}_" if self.IDENTIFIER else ""
        # Determine the base directory for saving results
        if hasattr(self, 'RESULTS_BASE_DIR') and self.RESULTS_BASE_DIR:
            base_dir = self.RESULTS_BASE_DIR
            results_dir = os.path.join(base_dir, f"{folder_name}")
        else:
            base_dir = "Results"  # Default directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_dir = os.path.join(base_dir, f"{folder_name}{timestamp}")

        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Save results as CSV
        results_csv_path = os.path.join(results_dir, "results.csv")
        with open(results_csv_path, "w") as csv_file:
            csv_file.write("Execution,BestFitness,BestSolution\n")
            for i, result in enumerate(results, start=1):
                csv_file.write(f"{i},{result['BestFitness']},{result['BestSolution']}\n")

        # Add ENABLE_FITNESS_FUNCTION_VISUALIZATION to the configuration
        config["ENABLE_FITNESS_FUNCTION_VISUALIZATION"] = self.ENABLE_FITNESS_FUNCTION_VISUALIZATION

        # Save configuration as JSON
        config["IDENTIFIER"] = self.IDENTIFIER
        config_json_path = os.path.join(results_dir, "config.json")
        with open(config_json_path, "w") as json_file:
            json.dump(config, json_file, indent=4)

        # Save figures as PNG
        for fig, filename in self.figures:
            fig_path = os.path.join(results_dir, filename)
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)  # Ensure the directory exists
            fig.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving

        # Save performance metrics as CSV
        metrics_csv_path = os.path.join(results_dir, "performance_metrics.csv")
        with open(metrics_csv_path, "w") as csv_file:
            csv_file.write("Metric,Value\n")
            for metric, value in performance_metrics.items():
                csv_file.write(f"{metric},{value}\n")
            
            # Add configuration section
            csv_file.write("\nCONFIGURATION\n")
            for key, value in config.items():
                csv_file.write(f"{key},{value}\n")

        # Save curve data with standard deviations as CSV
        for curve_name, data in curve_data.items():
            curve_csv_path = os.path.join(results_dir, f"{curve_name}.csv")
            with open(curve_csv_path, "w") as csv_file:
                csv_file.write("Generation,Value,StdDev\n")  # Header
                for i, (value, std) in enumerate(zip(data, curve_std_data[curve_name])):
                    csv_file.write(f"{i},{value},{std}\n")  # Data with standard deviation

        # Save optimal points distribution with standard deviation as CSV
        optimal_points_csv_path = os.path.join(results_dir, "optimal_points_distribution.csv")
        with open(optimal_points_csv_path, "w") as csv_file:
            csv_file.write("X,Y,X_Std,Y_Std\n")  # Header
            for point in optimal_points:
                csv_file.write(f"{point[0]},{point[1]},{optimal_points_std[0]},{optimal_points_std[1]}\n")

        # Save diversity metrics (standard deviation and Euclidean distance) as CSV
        diversity_csv_path = os.path.join(results_dir, "diversity_metrics.csv")
        with open(diversity_csv_path, "w") as csv_file:
            csv_file.write("Generation,StdDevDiversity,EuclideanDiversity\n")  # Header
            for generation, (std_dev, euclidean) in enumerate(zip(self.diversity_per_generation, self.euclidean_diversity_per_generation)):
                csv_file.write(f"{generation},{std_dev},{euclidean}\n")  # Data for each generation


        print(f"Results saved in: {results_dir}")

