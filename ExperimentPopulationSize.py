from MainOptimizationScript import MainOptimizationScript
import os
import pandas as pd
import matplotlib.pyplot as plt

def experiment_population_sizes(sizes, num_executions, fitness_function, identifier_prefix):
    """
    Run experiments with different population sizes and collect results.
    :param sizes: List of population sizes to test.
    :param num_executions: Number of executions for each population size.
    :param fitness_function: Fitness function to use.
    :param identifier_prefix: Prefix for result folders.
    """
    for size in sizes:
        print(f"Running experiment with POPULATION_SIZE={size}")
        OptimizationObject = MainOptimizationScript(FITNESS_FUNCTION_SELECTION=fitness_function, IDENTIFIER=f"{identifier_prefix}_POP{size}")
        OptimizationObject.POPULATION_SIZE = size
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=[1, 1])

def load_performance_metrics(results_dir):
    """
    Load performance metrics from all subdirectories in the results directory.
    :param results_dir: Path to the directory containing experiment results.
    :return: DataFrame with combined performance metrics.
    """
    data = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            metrics_path = os.path.join(folder_path, "performance_metrics.csv")
            if os.path.exists(metrics_path):
                try:
                    # Use on_bad_lines='skip' to ignore problematic lines
                    df = pd.read_csv(metrics_path, skip_blank_lines=True, on_bad_lines='skip')
                    df["Experiment"] = folder  # Add experiment identifier
                    data.append(df)
                except pd.errors.ParserError as e:
                    print(f"Error reading {metrics_path}: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def plot_fitness_vs_population(data):
    """
    Plot the best fitness vs. population size.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows where Metric is "Best Solution Found"
    fitness_data = data[data["Metric"] == "Best Solution Found"].copy()
    fitness_data.loc[:, "PopulationSize"] = fitness_data["Experiment"].str.extract(r'POP(\d+)').astype(int)
    avg_fitness = fitness_data.groupby("PopulationSize")["Value"].mean()
    
    plt.figure()
    avg_fitness.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Best Fitness vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Average Best Fitness")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("fitness_vs_population.png")
    plt.show()

def plot_execution_time_vs_population(data):
    """
    Plot the execution time vs. population size.
    :param data: DataFrame containing performance metrics.
    """
    data["PopulationSize"] = data["Experiment"].str.extract(r'POP(\d+)').astype(int)
    avg_time = data.groupby("PopulationSize")["Total Execution Time (s)"].mean()
    plt.figure()
    avg_time.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Execution Time vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Execution Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("execution_time_vs_population.png")
    plt.show()

def plot_convergence_curves(results_dir):
    """
    Plot convergence curves for each population size.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            convergence_path = os.path.join(folder_path, "convergence_curve.csv")
            if os.path.exists(convergence_path):
                df = pd.read_csv(convergence_path)
                generations = df["Generation"]
                fitness = df["Value"]
                plt.plot(generations, fitness, label=folder)
    plt.title("Convergence Curves")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(title="Population Size", loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("convergence_curves.png")
    plt.show()

def plot_fitness_std_vs_population(data):
    """
    Plot the standard deviation of the best fitness vs. population size.
    :param data: DataFrame containing performance metrics.
    """
    data["PopulationSize"] = data["Experiment"].str.extract(r'POP(\d+)').astype(int)
    std_fitness = data.groupby("PopulationSize")["Best Solution Found"].std()
    plt.figure()
    std_fitness.plot(kind="bar", color="salmon", edgecolor="black")
    plt.title("Standard Deviation of Best Fitness vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Fitness Standard Deviation")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("fitness_std_vs_population.png")
    plt.show()

def plot_average_diversity_vs_population(results_dir):
    """
    Plot the average diversity vs. population size.
    :param results_dir: Path to the directory containing experiment results.
    """
    diversity_data = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_curve.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                avg_diversity = df["Value"].mean()
                population_size = int(folder.split("_POP")[1].split("_")[0])
                diversity_data.append({"PopulationSize": population_size, "AverageDiversity": avg_diversity})
    if diversity_data:
        df_diversity = pd.DataFrame(diversity_data)
        plt.figure()
        df_diversity.set_index("PopulationSize")["AverageDiversity"].plot(kind="bar", color="purple", edgecolor="black")
        plt.title("Average Diversity vs. Population Size")
        plt.xlabel("Population Size")
        plt.ylabel("Average Diversity")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("average_diversity_vs_population.png")
        plt.show()

def plot_diversity_curves(results_dir):
    """
    Plot diversity curves for each population size.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_curve.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                generations = df["Generation"]
                diversity = df["Value"]
                plt.plot(generations, diversity, label=folder)
    plt.title("Diversity Curves")
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.legend(title="Population Size", loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("diversity_curves.png")
    plt.show()

def plot_success_rate_vs_population(data, optimal_solution, tolerance=1e-2):
    """
    Plot the success rate vs. population size.
    :param data: DataFrame containing performance metrics.
    :param optimal_solution: Known optimal solution.
    :param tolerance: Tolerance for determining success.
    """
    data["PopulationSize"] = data["Experiment"].str.extract(r'POP(\d+)').astype(int)
    data["Success"] = data["Chromosome for Best Solution"].apply(
        lambda x: pd.eval(x) if isinstance(x, str) else x
    ).apply(
        lambda sol: sum((a - b) ** 2 for a, b in zip(sol, optimal_solution)) ** 0.5 <= tolerance
    )
    success_rate = data.groupby("PopulationSize")["Success"].mean() * 100
    plt.figure()
    success_rate.plot(kind="bar", color="gold", edgecolor="black")
    plt.title("Success Rate vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("success_rate_vs_population.png")
    plt.show()

def main():
    # Define the population sizes to test
    population_sizes = [50, 100, 200, 500]

    # Run the experiments
    experiment_population_sizes(
        sizes=population_sizes,
        num_executions=10,
        fitness_function='Levi',
        identifier_prefix='LeviExperiment'
    )

    # Load results and generate plots
    results_dir = "Results"  # Path to the directory containing experiment results
    data = load_performance_metrics(results_dir)
    if data.empty:
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_population(data)
    plot_execution_time_vs_population(data)
    plot_convergence_curves(results_dir)
    plot_fitness_std_vs_population(data)
    plot_average_diversity_vs_population(results_dir)
    plot_diversity_curves(results_dir)
    plot_success_rate_vs_population(data, optimal_solution=[1, 1])

if __name__ == "__main__":
    main()