from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union, TypeVar
from torch import Tensor
from .torch_cpm.config import TorchCPMCellType
from pathlib import Path
import yaml
import numpy as np


class Guideline(BaseModel):
    type: Literal["circle"]
    x_center: float
    y_center: float
    radius: float
    radial_std: float = Field(1.0, description="Radial StD for sampling cell centroids")
    n_cells: Optional[int] = Field(5, description="Number of cell sampled in this guideline")
    tangent_distribution: Optional[Literal["uniform"]] = Field("uniform", description="Distribution along this guideline")
    radial_distribution: Optional[Literal["normal"]] = Field("normal", description="Distribution across this guideline")
    spawned_cell_ids: List[int] = Field(default=[], description="List of cell_id that has been successfully spawned on this guideline.")

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
    start_coordinates_cpm: List[float]
    start_coordinates: List[float]
    cell_type: TorchCPMCellType = Field(description="Assigned Cell Type of this cell")
    params: Optional[Union[CellParams, None]] = Field(default=None, description="Parameter array of this cell. Values from which the gene vector is produced.")
    

TTissue = TypeVar("TTissue", bound="Tissue")
class Tissue(BaseModel):
    """
        Represents a sampled tissue
    """
    id: int = Field(description="An identifier for this tissue within the current dataset.")
    tissue_seed: int = Field(description="Seed to set to reproduce this tissue given a SimConfig.")
    cpm_step: int = Field(description="Number of CMP simulation steps run")
    rd_step: int = Field(description="Number of RD simulation steps run")
    guidelines: List[Guideline] = Field(description="A list of Guideline objects describing the guideline used to spawn cells")
    cells: List[Cell] = Field([], description="Cells belonging to this tissue")
    cell_grid: Tensor = Field(description="Tensor containing the cell definitions (ID, CellType)")
    subcell_grid: Tensor = Field(description="Tensor containing the subcellular components")
    
    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def from_yaml(tissue_yaml_fp: Path) -> TTissue:
        """
            Read a tissue from its yaml file.
        """
        tissue_yaml_fp = Path(tissue_yaml_fp)
        tissue_cell_fp = Path(str(tissue_yaml_fp).replace(".yaml", "_cell.npy"))
        tissue_subcell_fp = Path(str(tissue_yaml_fp).replace(".yaml", "_subcell.npy"))

        with tissue_yaml_fp.open("r") as f:
            data = yaml.safe_load(f)
            data["cell_grid"] = Tensor(np.load(tissue_cell_fp)).int()
            data["subcell_grid"] = Tensor(np.load(tissue_subcell_fp)).float()
        return Tissue(**data)
    
    @staticmethod
    def from_folder(tissue_folder: Path) -> List[TTissue]:
        """
            Read a sequence of Tissues from a tissue folder.
        """
        tissue_folder = Path(tissue_folder)
        tissues_fp = sorted(list(tissue_folder.rglob("*.yaml")))
        tissues = []

        for tissue_yaml_fp in tissues_fp:
            tissues.append(Tissue.from_yaml(tissue_yaml_fp))
        return tissues

    def save(self, tissue_folder: Path):
        """
            Saves a tissue to a folder, storing the .yaml file with all the parameters and the grids as .npy files.
        """

        tissue_folder.mkdir(exist_ok=True, parents=True)

        tissue_id = self.id
        tissue_step = self.cpm_step
        
        out_path_yaml = tissue_folder.joinpath(f"t_{tissue_id:07d}_s_{tissue_step:07d}.yaml")
        out_path_cell = tissue_folder.joinpath(f"t_{tissue_id:07d}_s_{tissue_step:07d}_cell.npy")
        out_path_subcell = tissue_folder.joinpath(f"t_{tissue_id:07d}_s_{tissue_step:07d}_subcell.npy")
        
        with open(out_path_yaml, "w") as file:
            yaml.dump(self.model_dump(exclude={"cell_grid", "subcell_grid"}), file)

        np.save(out_path_cell, self.cell_grid.cpu().numpy())
        np.save(out_path_subcell, self.subcell_grid.cpu().numpy())