import os
import subprocess

def test_experiment_scripts():
    """
    Test all experiment scripts in the Experiments_1A folder.
    """
    # Define the directory containing the experiment scripts
    experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Experiments_1A"))

    # Iterate through all subdirectories and files in the Experiments_1A folder
    for root, _, files in os.walk(experiments_dir):
        for file in files:
            # Check if the file is a Python script and matches the experiment naming convention
            if file.endswith(".py") and file.startswith("Experiment"):
                script_path = os.path.join(root, file)
                print(f"Testing script: {script_path}")

                # Try to execute the script using subprocess
                try:
                    result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
                    print(f"Output of {file}:")
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error while running {file}:")
                    print(e.stderr)
                except Exception as e:
                    print(f"Unexpected error while running {file}: {e}")

if __name__ == "__main__":
    test_experiment_scripts()