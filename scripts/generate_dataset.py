from multiomicscellsim import Simulator
from multiomicscellsim.config import SimulatorConfig

from pathlib import Path
import argparse

def generate_dataset(config_path: Path):
    """
    Generate a dataset using the specified simulator configuration.

    Args:
        config_path (Path): Path to the YAML configuration file.
    """
    sim_config = SimulatorConfig.from_yaml(config_path)
    sim = Simulator(config=sim_config)
    print(f"Running simulation from config: {config_path}")
    tissues = sim.sample()
    
    # Save or process the generated tissues if needed
    print(f"Generated {len(tissues)} tissue(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets using the Mycroverse.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file."
    )
    
    args = parser.parse_args()
    
    # Convert the config path to a Path object and call the main function
    config_path = Path(args.config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    generate_dataset(config_path)