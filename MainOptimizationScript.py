from Library.SelectionMethods import SelectionMethods
from Library.CrossoverMethods import CrossoverMethods
from Library.MutationMethods import MutationMethods
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
import time  # Import for execution time measurement

class MainOptimizationScript:
    def __init__(self, FITNESS_FUNCTION_SELECTION):
        """
        Constructor to initialize the class attributes.
        """
        # Class Parameters
        self.ENABLE_FITNESS_FUNCTION_VISUALIZATION = False
        self.ALLOWED_FITNESS_FUNCTIONS = ['Base', 'Akley', 'Custom']
        self.ResultsOverall = []  # Store performance data for all executions
        self.BestResult = None   # Store the best execution result
        self.diversity_per_generation = []  # Store diversity metrics
        self.best_fitness_per_generation = []  # Store best fitness per generation

        # Validate FITNESS_FUNCTION_SELECTION
        if FITNESS_FUNCTION_SELECTION not in self.ALLOWED_FITNESS_FUNCTIONS:
            raise ValueError(f"Invalid FITNESS_FUNCTION_SELECTION. Allowed values are: {self.ALLOWED_FITNESS_FUNCTIONS}")

        # Initialize configuration parameters
        self.POPULATION_SIZE = 100
        self.GENERATION_COUNT = 50
        
        self.CHROMOSOME_LENGTH = 2
        self.LOWER_BOUND = -100
        self.UPPER_BOUND = 100
        self.FITNESS_FUNCTION_SELECTION = FITNESS_FUNCTION_SELECTION
        self.SELECTION_METHOD = 'TournamentSelection'
        self.SELECTION_TOURNAMENT_SIZE = 10
        self.CROSSOVER_METHOD = 'SinglePointCrossover'
        self.CROSSOVER_RATE = 0.8
        self.MUTATION_METHOD = 'RandomMutationOnIndividualGenes'
        self.MUTATION_RATE = 0.5
        self.OPTIMIZATION_METHOD = 'Elitism'
        self.OPTIMIZATION_METHOD_NUMBER_ELITES = 10



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
            case _:
                raise ValueError("Invalid FITNESS_FUNCTION_SELECTION")
        return fitness_value
        
    def single_optimization(self):
        """
        Initialize the single optimization process.
        """
        self.visualize_fitness_function()

        self.elitism_optimization()


    def multiple_optimization(self, num_executions, optimal_solution=None, tolerance=1e-1):
        """
        Evaluate the performance of the optimization algorithm.
        :param num_executions: Number of times to run the optimization.
        :param optimal_solution: Known optimal solution value (if available).
        :param tolerance: Tolerance for determining success in finding the optimal solution.
        """
        success_count = 0
        best_solutions = []
        best_fitness_values = []
        best_chromosomes = []  # Store the best chromosomes

        self.visualize_fitness_function()
        self.ResultsOverall = []  # Reset results for new executions
        best_overall_fitness = float('inf')  # Initialize best fitness as infinity

        all_best_fitness_per_generation = []  # Store best fitness per generation for all executions
        all_diversity_per_generation = []  # Store diversity per generation for all executions

        start_time = time.time()  # Start timing
        for execution in range(1, num_executions + 1):
            self.elitism_optimization()
            # Retrieve the best solution and its fitness value from the last generation
            best_individual = min(self.population_fitness, key=lambda x: x[1])
            best_solution = best_individual[0]
            best_fitness = best_individual[1]

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

            # Print progress information
            elapsed_time = time.time() - start_time
            print(f"Execution {execution}/{num_executions} completed. Best Fitness: {best_fitness:.6f}. Elapsed Time: {elapsed_time:.2f} seconds")

            best_solutions.append(best_individual[0])
            best_fitness_values.append(best_individual[1])
            best_chromosomes.append(best_individual[0])  # Store the best chromosome

            # Check if the solution is within the tolerance of the optimal solution
            if optimal_solution is not None:
                # Calculate Euclidean distance between the solution and the optimal solution
                distance = sqrt(sum((a - b) ** 2 for a, b in zip(best_individual[0], optimal_solution)))
                if distance <= tolerance:
                    success_count += 1

            all_best_fitness_per_generation.append(self.best_fitness_per_generation)
            all_diversity_per_generation.append(self.diversity_per_generation)

        # Calculate performance metrics
        avg_best_fitness = sum(result["BestFitness"] for result in self.ResultsOverall) / num_executions
        success_count = sum(
            1 for result in self.ResultsOverall
            if optimal_solution is not None and sqrt(sum((a - b) ** 2 for a, b in zip(result["BestSolution"], optimal_solution))) <= tolerance
        )
        success_rate = success_count / num_executions
        best_overall_solution = min(best_fitness_values)
        best_overall_chromosome = best_chromosomes[best_fitness_values.index(best_overall_solution)]

        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"Total Execution Time: {execution_time:.2f} seconds")

        print(f"Performance Metrics:")
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Average Best Fitness: {avg_best_fitness:.6f}")
        print(f"Best Solution Found: {self.BestResult['BestFitness']}")
        print(f"Chromosome for Best Solution: {self.BestResult['BestSolution']}")

        # Calculate aggregated metrics
        avg_best_fitness_per_generation = np.mean(all_best_fitness_per_generation, axis=0)
        std_best_fitness_per_generation = np.std(all_best_fitness_per_generation, axis=0)
        avg_diversity_per_generation = np.mean(all_diversity_per_generation, axis=0)
        std_diversity_per_generation = np.std(all_diversity_per_generation, axis=0)

        # Plot aggregated metrics
        self.plot_convergence_curve(avg_best_fitness_per_generation, std_best_fitness_per_generation)
        self.plot_population_diversity(avg_diversity_per_generation, std_diversity_per_generation)

    def elitism_optimization(self):
        population = [self.generate_chromosome() for _ in range(self.POPULATION_SIZE)]
        #Optimization loop

        self.diversity_per_generation = []  # Reset diversity metrics
        self.best_fitness_per_generation = []  # Reset best fitness metrics
        for idx in range(self.GENERATION_COUNT):
            #Evaluate fitness of the population
            population_fitness = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in population]
            self.population_fitness = population_fitness
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
            best_individual = min(population_fitness, key=lambda x: x[1])
            best_individual_value =  best_individual [0] 
            best_individual_fitness_value = best_individual[1] # Evaluate fitness of the best individual

            # Update population
            population = [elite[0] for elite in elites] + [chromosome[0] for chromosome in new_population_fitness]

            # Calculate and store diversity
            diversity = np.std([chromosome for chromosome, _ in population_fitness])
            self.diversity_per_generation.append(diversity)
            # Store best fitness for convergence curve
            best_fitness = min(population_fitness, key=lambda x: x[1])[1]
            self.best_fitness_per_generation.append(best_fitness)
    



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

        selected_parents_ = []
        # Selection Strategies
        match self.SELECTION_METHOD:
        # 1. Tournament Selection
            case 'TournamentSelection':
                  # List to store selected parents
                selected_parents = SelectionMethods.tournament_selection(population_fitness, tournament_size=self.SELECTION_TOURNAMENT_SIZE, selected_parents=selected_parents_)
            case _:
                selected_parents = selected_parents_
                raise ValueError("Invalid SELECTION_METHOD")                
        return selected_parents  # Return the list of selected parents

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        """
        # Crossover Strategies
        if random.random() < self.CROSSOVER_RATE:
            match self.CROSSOVER_METHOD:
            # 1. Single-point crossover
                case 'SinglePointCrossover':
                    child1, child2 = CrossoverMethods.single_point_crossover(parent1, parent2)
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
            case _:
                raise ValueError("Invalid MUTATION_METHOD")
        return mutated_individual

    def visualize_fitness_function(self):
        """
        Visualize the fitness function in 3D.
        """
        if self.ENABLE_FITNESS_FUNCTION_VISUALIZATION:
            print("Fitness function visualization is disabled.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x = np.linspace(self.LOWER_BOUND, self.UPPER_BOUND, 100)
        y = np.linspace(self.LOWER_BOUND, self.UPPER_BOUND, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate_fitness([X, Y])
        ax.plot_surface(X, Y, Z, cmap='viridis')    
        plt.show(block=False)  # Allow script to continue
        plt.pause(0.1)  # Ensure the figure is rendered

    def plot_convergence_curve(self, avg_best_fitness, std_best_fitness):
        """
        Plot the aggregated convergence curve showing the average best fitness per generation.
        """
        plt.figure()
        generations = range(len(avg_best_fitness))
        plt.plot(generations, avg_best_fitness, label="Average Best Fitness")
        plt.fill_between(generations, 
                         avg_best_fitness - std_best_fitness, 
                         avg_best_fitness + std_best_fitness, 
                         color='blue', alpha=0.2, label="Std Dev")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Aggregated Convergence Curve")
        plt.legend()
        plt.show(block=False)  # Allow script to continue
        plt.pause(0.1)  # Ensure the figure is rendered

    def plot_population_diversity(self, avg_diversity, std_diversity):
        """
        Plot the aggregated diversity of the population over generations.
        """
        plt.figure()
        generations = range(len(avg_diversity))
        plt.plot(generations, avg_diversity, label="Average Diversity")
        plt.fill_between(generations, 
                         avg_diversity - std_diversity, 
                         avg_diversity + std_diversity, 
                         color='orange', alpha=0.2, label="Std Dev")
        plt.xlabel("Generation")
        plt.ylabel("Diversity (Standard Deviation)")
        plt.title("Aggregated Population Diversity")
        plt.legend()
        plt.show(block=False)  # Allow script to continue
        plt.pause(0.1)  # Ensure the figure is rendered