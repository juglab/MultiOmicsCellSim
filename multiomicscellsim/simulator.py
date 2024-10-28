import json
import logging
from pathlib import Path

from .config import SimulatorConfig

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

    def __init__(self, config_fp=None):
        # Load the configuration or use default values
        self.config = SimulatorConfig() if config_fp is None else self.load_config(config_fp)

    def load_config(self, config_fp):
        try:
            with open(config_fp, 'r') as f:
                config_data = json.load(f)
            logger.setLevel(LOG_LEVELS[config_data["debug_level"]]) 
            return SimulatorConfig(**config_data)  # Validate and create an instance
        
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def run(self):
        logging.info(f"Running simulator with configuration: {self.config.json()}")

    def generate_default_json(self):
        """
            Generate a json containing default parameters as starting point for creating a custom one.
        """
        return self.config.json()


