from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import torch
from functools import cached_property
from multiomicscellsim.patterns.config import RDPatternConfig, ReactionDiffusionConfig

class TorchCPMCellType(BaseModel):
    id: int = Field(1, description="Unique Identifier for the cell type. Cannot be 0 for consistency with pixel values")
    name: Optional[str] = Field("", description="Human Readable name of the cell type")
    
    # Constraint-specific parameters
    background_adhesion: float=Field(10.0, description="Adhesion penalty term with the background for this cell type. Used with AdhesionConstraint. Higher values will make the cell to minimize contacts with the background (be rounder) but it can also make it disappear if no opposing constraints are in place")
    cells_adhesion: torch.Tensor = Field(None, description="Adhesion energy with other cell types, excluding the background. Used with AdhesionConstraint. Higher values will make the cell avoid contacts with others, lower values will make the cell stick with others.")
    
    preferred_volume: int = Field(None, description="Preferred volume for this cell type. Used with VolumeConstraint.")
    preferred_local_perimeter: int = Field(8, description="Preferred local perimeter for this cell type. Used with LocalPerimeterConstraint. 3: edges try to be flat and either vertical or horizontal. Greater numbers increases rougness or cuvature")
    subcellular_pattern: Union[None, RDPatternConfig] = Field(None, description="Subcellular pattern to use for this cell type. If None, no subcellular simulation will be run for this cell type.")
    
    class Config:
        arbitrary_types_allowed = True

class TorchCPMConfig(BaseModel):
    size: int = Field(256, description="Size of the grid for the CPM simulation.")
    frontier_probability: float = Field(0.1, description="Probability for a pixel to be selected for copy attempt.")
    temperature: float = Field(0.1, description="Temperature for the algorithm.")

    cell_types: List[TorchCPMCellType] = Field([], description="List of cell types for the simulation, excluding the background. Indices should start at 1.")

    lambda_volume: float = Field(1.0, description="Weight for the volume energy.")
    lambda_perimeter: float = Field(1.0, description="Weight for the perimeter energy.")
    
    run_rd_every: int = Field(100, description="Number of steps to run the reaction diffusion simulation.")
    rd_steps: int = Field(10000, description="Number of steps to run the reaction diffusion simulation.")


    class Config:
        arbitrary_types_allowed = True

    @cached_property
    def adhesion_matrix(self):
        """
            Return the adhesion matrix for the cell types.
            NOTICE: THIS IS A CACHED PROPERTY, THIS WILL PREVENT THE PARAMETER TO BE CHANGED OVER TIME
        """
        adhesion_matrix = torch.zeros(len(self.cell_types)+1, len(self.cell_types)+1)
        
        for i, cell_type in enumerate([None] + self.cell_types):
            for j, other_cell_type in enumerate([None] + self.cell_types):
                if i == 0 and j == 0:
                    adhesion_matrix[i, j] = 0
                elif i == 0:
                    # Source is background, take the background adhesion from j
                    adhesion_matrix[i, j] = other_cell_type.background_adhesion
                elif j == 0:
                    adhesion_matrix[i, j] = cell_type.background_adhesion
                else:
                    adhesion_matrix[i, j] = cell_type.cells_adhesion[j-1]
        return adhesion_matrix

    
    @cached_property
    def preferred_volumes(self):
        """
            Return the preferred volumes for the cell types
            NOTICE: THIS IS A CACHED PROPERTY, THIS WILL PREVENT THE PARAMETER TO BE CHANGED OVER TIME
        """
        return torch.tensor([cell_type.preferred_volume for cell_type in self.cell_types])
    
    @cached_property
    def preferred_local_perimeters(self):
        """
            Return the preferred local perimeters for the cell types
            NOTICE: THIS IS A CACHED PROPERTY, THIS WILL PREVENT THE PARAMETER TO BE CHANGED OVER TIME
        """
        return torch.tensor([cell_type.preferred_local_perimeter for cell_type in self.cell_types])
    