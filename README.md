# Evolutional Computing Repository

## Description
This repository contains the implementation of genetic algorithms for optimization, including selection, crossover, and mutation methods. The main script, `MainOptimizationScript.py`, is responsible for executing the optimization process and evaluating the algorithm's performance.

## Repository Structure
```
EvolutionalComputing/
├── MainOptimizationScript.py       # Main script for optimization
├── Playground.py                   # Auxiliary script for testing and experimentation
├── Library/                        # Library with selection, crossover, and mutation methods
│   ├── CrossoverMethods.py         # Crossover methods
│   ├── MutationMethods.py          # Mutation methods
│   ├── SelectionMethods.py         # Selection methods
│   └── __pycache__/                # Compiled files
├── Results/                        # Directory for storing results
└── _compiled/                      # Directory for compiled files
```

## Main Scripts

### MainOptimizationScript.py
This is the main script that implements the genetic algorithm. It includes:
- **Algorithm Configuration**: Parameters such as population size, number of generations, selection, crossover, and mutation methods.
- **Fitness Functions**: Implementation of fitness functions like Base, Ackley, Drop-Wave, and Levi.
- **Performance Metrics**: Success rate, best fitness, population diversity, and execution time.
- **Visualizations**: Convergence, population diversity, and optimal points distribution plots.

#### Key Methods
- `evaluate_fitness`: Evaluates the fitness of a chromosome based on the selected function.
- `single_optimization`: Executes a single optimization.
- `multiple_optimization`: Executes multiple optimizations and calculates aggregated metrics.
- `elitism_optimization`: Implements the elitism method for population evolution.
- `plot_convergence_curve`: Plots the convergence curve.
- `plot_population_diversity`: Plots the population diversity.
- `plot_optimal_points`: Plots the distribution of optimal points.

#### Execution Flow
Below is a detailed flowchart of the genetic algorithm's execution steps:

```plaintext
Start
 |
 v
Initialize Population
 |
 v
Evaluate Fitness of Population
 |
 v
For Each Generation:
    |
    v
    Select Parents
        - Use the configured selection method (e.g., Tournament Selection)
        - Select individuals based on fitness
    |
    v
    Perform Crossover
        - Use the configured crossover method (e.g., Single-Point Crossover)
        - Combine parent chromosomes to produce offspring
    |
    v
    Apply Mutation
        - Use the configured mutation method (e.g., Random Mutation on Individual Genes)
        - Introduce random changes to offspring chromosomes
    |
    v
    Evaluate Fitness of Offspring
        - Calculate fitness for each offspring chromosome
    |
    v
    Apply Elitism (Retain Best Individuals)
        - Retain a specified number of the best individuals from the current population
 |
 v
End Generations
 |
 v
Calculate Metrics (e.g., Best Fitness, Diversity)
    - Aggregate metrics across all generations
    - Calculate success rate, average fitness, and diversity
 |
 v
Save Results and Visualizations
    - Save performance metrics, configuration, and plots
 |
 v
End
```

### Playground.py
This script is used for testing and experimenting with different configurations and methods of the genetic algorithm.

### Library/
Contains implementations of auxiliary methods:
- `SelectionMethods.py`: Selection methods, such as tournament selection.
- `CrossoverMethods.py`: Crossover methods, such as single-point crossover.
- `MutationMethods.py`: Mutation methods, such as random mutation on individual genes.

## How to Use
1. Configure the parameters in the `MainOptimizationScript.py` file.
2. Run the script to perform optimizations:
   ```bash
   python MainOptimizationScript.py
   ```
3. Visualize the results in the generated plots and metrics displayed in the console.

## Requirements
- Python 3.11 or higher
- Required libraries:
  - `numpy`
  - `matplotlib`

Install the dependencies with:
```bash
pip install numpy matplotlib
```

## Contact
For questions or suggestions, contact Gabriel Fernandes.