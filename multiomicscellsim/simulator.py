import json
import logging
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation



from .config import SimulatorConfig
from .entities import Tissue

from .tissue_generator import TissueGenerator

import random
import os
import torch
import numpy as np
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level
logger = logging.getLogger(__name__)      # Create a logger instance

# Mapping of debug levels to logging constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}

class Simulator:
    config: SimulatorConfig

    def __init__(self, config: SimulatorConfig, config_fp=None):
        # Load the configuration or use default values
        if config is not None:
            self.config = config
            logger.debug(f"Loaded configuration from instance")
        else:
            if config_fp is None:
                logger.warning("No configuration provided. Using default values.")
                self.config = SimulatorConfig()
            else:
                self.config = self.load_config(config_fp)
                logger.debug(f"Loaded configuration from file: {config_fp}")
        
        # Set seed if not specified in config
        self._set_simulator_seed()
        # Apply simulator-wise seed
        self._apply_seed(self.config.simulator_seed)


    def _set_simulator_seed(self):
        """
            Checks the simulator seed in the configuration. 
            If None, generate a random seed to use, to ensure reproducible results.
        """
        if self.config.simulator_seed is None:
            self.config.simulator_seed = random.randint(0, 2**32 - 1)
            logger.info(f"Simulator seed has ben set to {self.config.simulator_seed}")
        return self.config.simulator_seed

    def _apply_seed(self, seed=None):
        """Apply the seed to all randomness sources."""
        
        logger.debug(f"Setting random seed to {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)  # Enforce deterministic behavior
        os.environ["OMP_NUM_THREADS"] = "1"  # Single-threaded execution
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_config(self, config_fp):
        try:
            with open(config_fp, 'r') as f:
                config_data = json.load(f)
            logger.setLevel(LOG_LEVELS[config_data["debug_level"]]) 
            return SimulatorConfig(**config_data)  # Validate and create an instance
        
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_root_folder(self):
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def _get_dataset_folder(self):
        """
            Get the first dataset folder available by increasing the dataset index iteratively
        """
        dataset_idx = 0
        dataset_folder = self.config.output_root.joinpath(f"{self.config.dataset_prefix}{dataset_idx}")

        while dataset_folder.exists():
            dataset_idx += 1
            dataset_folder = self.config.output_root.joinpath(f"{self.config.dataset_prefix}{dataset_idx}")
        
        dataset_folder.mkdir(parents=True, exist_ok=True)
        return dataset_folder
    
    def dump_config(self, path: Path):
        with open(path, "w") as file:
            config_dict = self.config.model_dump()
            # Yaml does not manage Path objects
            config_dict["output_root"] = str(config_dict["output_root"])
            yaml.dump(config_dict, file)
        logger.info(f"Simulation Configuration written to {path}")

    def log_error_seed(self, path: Path, seed: int):
        """
            Log seeds that failed generation in a path for debugging
        """
        with open(path, "+a") as debug_file:
            debug_file.write(f"{seed}\n")


    def sample(self):
        """
            Generate a new tissue sample.
        """

        dataset_folder = self._get_dataset_folder()
        
        self.dump_config(dataset_folder.joinpath("sim_config.yaml"))

        tissue_seeds = [random.randint(0, 2**32 - 1) for _ in range(self.config.n_simulations)]

        tissue_generator = TissueGenerator(simulator_config=self.config)

        tissues = list()
        for tissue_id, tissue_seed in enumerate(tissue_seeds):
            try:
                tissue_folder = dataset_folder.joinpath(self.config.tissue_folder, str(tissue_id))

                # Set the seed for the current tissue
                self._apply_seed(tissue_seed)
                tissue_steps = tissue_generator.sample(tissue_id=tissue_id,
                                                            tissue_folder=tissue_folder,
                                                            seed=tissue_seed)
                tissues.append(tissue_steps)
            except:
                logger.error(f"Generation of tissue {tissue_id} with seed {tissue_seed} failed. Dumping seed to file.")
                self.log_error_seed(dataset_folder.joinpath("failed_seeds.log"), seed=tissue_seed)
        return tissues
    
    def plot_tissue(self, t: Tissue, axs=None):
        if axs is None:
            fig, axs = plt.subplots(2, 2)
            
        for i, ax in enumerate(axs.flatten()):
            ax.clear()
            if i == 0:
                ax.imshow(t.cell_grid[0], cmap='viridis')
                ax.set_title('Cell ID')
            elif i == 1:
                ax.imshow(t.cell_grid[1], cmap='viridis')
                ax.set_title('Cell Type')
            elif i == 2:
                ax.imshow(t.subcell_grid[0], cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Subcellular A')
            elif i == 3:
                ax.imshow(t.subcell_grid[1], cmap='viridis', vmin=0, vmax=1)
                ax.set_title('Subcellular B')

        if axs is None:
            fig.tight_layout()
            return fig
        else:
            return axs

    def plot_tissues(self, tl: List[Tissue]):
        fig, axs = plt.subplots(2, 2)
        update = lambda frame, tiss_list=tl, axes=axs: self.plot_tissue(tiss_list[frame], axs=axes)
        fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=len(tl), repeat=False)
        return ani.to_jshtml()

