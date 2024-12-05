from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Union

from .cpm.cpmentities import CPMCellType
from .cpm.constraints import HamiltonianConstraint

class MicroscopySpaceConfig(BaseModel):
    coord_min: float = 0.0
    coord_max: float = Field(1024, description="Maximum size of the image in micrometers")
    resolution: float = Field(1.0, description="Resolution in micrometers (How many um are represented in a pixel)")
    cpm_grid_size: int = Field(128, description="Grid size for the CPM simulation Grid")


class TissueConfig(BaseModel):

    # Guidelines
    n_curves: int = Field(2, description="Number of different guidelines to grow cells onto")
    curve_types: Literal["circle"] = Field("circle", description="Kind of guidelines to use")
    min_radius_perc: float = Field(0.25, description="Minimum radius for a circular guideline, in percentage of the image size")
    max_radius_perc: float = Field(0.3, description="Maximum radius for a circular guideline, in percentage of the image size")
    # TODO: Consider if everything should be in um or in %.
    guidelines_std: float = Field(50, description="List of standard deviation (in micrometers) across each point of each guideline")
    allow_guideline_intersection: bool = Field(False, description="Allow guidelines to intersect each other by removing any constraints on center sampling")

    cell_number_mean: int = Field(5, description="Average number of cells that are sampled on each guideline")
    cell_number_std: int = Field(3, description="Standard deviation of number of cells that are sampled on each guideline")
    
    cpm_temperature: float = Field(0.1, description="Temperature of the CPM simulation")
    cpm_iterations: int = Field(100, description="Number of iterations for the CPM simulation")
    cpm_cell_types: List[CPMCellType] = Field([], description="List of cell types to use in the CPM simulation. Differentiate the cell behaviour during growth")
    cpm_lambda_perimeter: float = Field(1.0, description="Scaling factor for the CPM perimeter energy")
    cpm_lambda_volume: float = Field(1.0, description="Scaling factor for the CPM volume energy")
    cpm_constraints: List[HamiltonianConstraint] = Field([], description="List of constraints to apply to the CPM simulation. Define the behaviour of the cells during growth")
class CellConfig(BaseModel):
    pass

class SubcellularConfig(BaseModel):
    pass


class SimulatorConfig(BaseModel):
    log_level: Literal["INFO", "WARNING", "DEBUG", "ERROR"] = Field("DEBUG", description="Logging Level")
    microscopy_space_config: MicroscopySpaceConfig = Field(MicroscopySpaceConfig(), description="Settings for the Microscopy modality")
    tissue_config: TissueConfig = Field(TissueConfig(), description="Tissue-level input features")
    cell_config: CellConfig = Field(CellConfig(), description="Cell-level input features")
    subcellular_config: SubcellularConfig = Field(SubcellularConfig(), description="Subcellular-level input features")