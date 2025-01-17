from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from torch import Tensor
from .torch_cpm.config import TorchCPMCellType

class Guideline(BaseModel):
    type: Literal["circle"]
    x_center: float
    y_center: float
    radius: float
    radial_std: float = Field(1.0, description="Radial StD for sampling cell centroids")
    n_cells: Optional[int] = Field(5, description="Number of cell sampled in this guideline")
    tangent_distribution: Optional[Literal["uniform"]] = Field("uniform", description="Distribution along this guideline")
    radial_distribution: Optional[Literal["normal"]] = Field("normal", description="Distribution across this guideline")

    def get_cell_type(self) -> int:
        pass

class CellParams(BaseModel):
    f: float = Field(description="Current feed ratio of this cell")
    k: float = Field(description="Current kill ratio of this cell")
    d_a: float = Field(description="Current diffusion term of A subcellular layer")
    d_b: float = Field(description="Current diffusion term of B subcellular layer")
    a_avg: float = Field(description="Average value of A layer")
    a_std: float = Field(description="Standard Deviation for tha A subcellular layer content")
    b_avg: float = Field(description="Average value of B layer")
    b_std: float = Field(description="Standard Deviation for tha B subcellular layer content")

class Cell(BaseModel):
    """
        Represents a sampled cell
    """
    cell_id: int = Field(description="Integer ID of the cell within the tissue. IDs are mapped to the first channel of the grid.")
    start_coordinates: List[float]
    cell_type: TorchCPMCellType = Field(description="Assigned Cell Type of this cell")
    params: Optional[Union[CellParams, None]] = Field(default=None, description="Parameter array of this cell. Values from which the gene vector is produced.")
    

class Tissue(BaseModel):
    """
        Represents a sampled tissue
    """
    cpm_step: int = Field(description="Number of CMP simulation steps run")
    rd_step: int = Field(description="Number of RD simulation steps run")
    guidelines: List[Guideline]
    cells: List[Cell] = Field([], description="Cells belonging to this tissue")
    cell_grid: Tensor = Field(description="Tensor containing the cell definitions (ID, CellType)")
    subcell_grid: Tensor = Field(description="Tensor containing the subcellular components")
    
    class Config:
        arbitrary_types_allowed = True