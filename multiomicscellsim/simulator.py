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
        self._apply_seed()
        self.tissue_generator = TissueGenerator(simulator_config=self.config)

    def _apply_seed(self):
        """Apply the seed to all randomness sources."""
        if self.config.seed is not None:
            seed = self.config.seed 
        else:
            seed = random.randint(0, 2**32 - 1)
            self.config.seed = seed
        logger.info(f"Setting random seed to {seed}")
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

    def generate_default_json(self):
        """
            Generate a json containing default parameters as starting point for creating a custom one.
        """
        return self.config
    
    def plot_debug(self, tissue: Tissue):
        self.tissue_generator.plot_debug(tissue)
        tissue.cpm_grid.render(0)
        tissue.cpm_grid.render(1)

    def sample(self, n=1):
        """
            Generate a new tissue sample.
        """
        # TODO: allow batched sampling using threadpools

        return self.tissue_generator.sample()
    
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

