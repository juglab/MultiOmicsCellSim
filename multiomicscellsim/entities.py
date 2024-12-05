from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

from .cpm.cpmentities import CPMCell, CPMGrid

class SampledEntity(BaseModel):
    seed: int
    def __init__(self, seed=None):
        self.seed = seed



class Guideline(BaseModel):
    type: Literal["circle"]
    x_center: float
    y_center: float
    radius: float
    radial_std: float = Field(1.0, description="Radial StD for sampling cell centroids")
    n_cells: Optional[int] = Field(5, description="Number of cell sampled in this guideline")
    tangent_distribution: Optional[Literal["uniform"]] = Field("uniform", description="Distribution along this guideline")
    radial_distribution: Optional[Literal["normal"]] = Field("normal", description="Distribution across this guideline")



class Cell(BaseModel):
    """
        Represents a sampled cell
    """
    start_coordinates: List[float]
    cpm_cell: CPMCell


class Tissue(BaseModel):
    """
        Represents a sampled tissue
    """

    guidelines: List[Guideline]
    cells: List[Cell] = Field([], description="Cells belonging to this tissue")
    cpm_grid: Union[CPMGrid, None] = Field(None, description="CPM Grid for the tissue")

class SubcellularShape():
    pass