from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class SampledEntity(BaseModel):
    seed: int
    def __init__(self, seed=None):
        self.seed = seed

class Guideline(BaseModel):
    type: Literal["circle"]
    x_center: float
    y_center: float
    radius: float
    variance: float
    n_cells: Optional[int] = Field(5, description="Number of cell sampled in this guideline")
    tangent_distribution: Optional[Literal["uniform"]] = Field("uniform", description="Distribution along this guideline")
    radial_distribution: Optional[Literal["normal"]] = Field("normal", description="Distribution across this guideline")

    

class Cell(BaseModel):
    """
        Represents a sampled cell
    """
    pass


class Tissue(BaseModel):
    """
        Represents a sampled tissue
    """

    guidelines: List[Guideline]
    cells: List[Cell] = Field([], description="Cells belonging to this tissue")




class SubcellularShape():
    pass