import json
import logging
from pathlib import Path

from .config import SimulatorConfig
from .entities import Tissue

from .tissue_generator import TissueGenerator

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

        self.tissue_generator = TissueGenerator(simulator_config=self.config)

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
    
