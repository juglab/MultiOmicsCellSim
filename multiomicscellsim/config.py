from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Union

from .torch_cpm.config import TorchCPMCellType
from .torch_cpm.constraints import TorchCPMConstraint
from multiomicscellsim.torch_cpm.config import TorchCPMConfig
from pathlib import Path

class MicroscopySpaceConfig(BaseModel):
    coord_min: float = 0.0
    coord_max: float = Field(1024, description="Maximum size of the image in micrometers")
    resolution: float = Field(1.0, description="Resolution in micrometers (How many um are represented in a pixel)")


class TissueConfig(BaseModel):

    # Guidelines
    n_curves: int = Field(2, description="Number of different guidelines to grow cells onto")
    curve_types: Literal["circles"] = Field("circles", description="Kind of guidelines to use")
    min_radius_perc: float = Field(0.25, description="Minimum radius for a circular guideline, in percentage of the image size")
    max_radius_perc: float = Field(0.3, description="Maximum radius for a circular guideline, in percentage of the image size")
    
    # TODO: Consider if everything should be in um or in %.
    guidelines_std: float = Field(50, description="List of standard deviation (in micrometers) across each point of each guideline")
    allow_guideline_intersection: bool = Field(False, description="Allow guidelines to intersect each other by removing any constraints on center sampling")

    cell_number_mean: int = Field(5, description="Average number of cells that are sampled on each guideline")
    cell_number_std: int = Field(3, description="Standard deviation of number of cells that are sampled on each guideline")

    cell_type_probabilities: List[List[float]] = Field([], description="Probability of each Guideline to spawn a cell type of the given type")
    
    initial_cell_size: int = Field(7, description="Initial Cell Size in pixels")
    

class SimulatorConfig(BaseModel):
    log_level: Literal["INFO", "WARNING", "DEBUG", "ERROR"] = Field("DEBUG", description="Logging Level")

    simulator_seed: Union[int, None] = Field(None, description="Random seed for the Simulator. This is used to generate individual seeds for each tissue and when sampling simulation-level parameters.")
    microscopy_space_config: MicroscopySpaceConfig = Field(MicroscopySpaceConfig(), description="Settings for the Microscopy modality")
    tissue_config: TissueConfig = Field(TissueConfig(), description="Tissue-level input features")
    cpm_config: TorchCPMConfig = Field(description="Configuration for the algorithm used for growing Cells (CPM / RD)")
    save_tissue_every: int = Field(0, description="Save / Return a Tissue state every N steps. If 0, only the last tissue is returned for each simulation. Last tissue is always returned.")
    output_root: Path = Field(description="Root Folder for dataset generation.")
    dataset_prefix: str = Field(default="mycroverse_", description="Prefix for each generated dataset. An ordinal number will be appended if a folder with the same name already exists.")
    tissue_folder: str = Field(default="tissues", description="Name of the subfolder containing tissues")
    n_simulations: int = Field(description="How many tissues to generate")