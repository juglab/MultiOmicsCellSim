from pydantic import BaseModel, Field
from typing import List, Dict
import torch


class TorchCPMConfig(BaseModel):
    size: int = Field(256, description="Size of the grid for the CPM simulation.")
    frontier_probability: float = Field(0.1, description="Probability for a pixel to be selected for copy attempt.")
    temperature: float = Field(0.1, description="Temperature for the algorithm.")
    adhesion_matrix: torch.Tensor = Field(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), description="Adhesion matrix for the algorithm.")
    preferred_volumes: torch.Tensor = Field(torch.tensor([100.0, 100.0]), description="Preferred volumes for each cell type.")
    preferred_perimeters: torch.Tensor = Field(torch.tensor([3.0, 3.0]), description="Preferred local perimeters for each cell type.")
    lambda_volume: float = Field(1.0, description="Weight for the volume energy.")
    lambda_perimeter: float = Field(1.0, description="Weight for the perimeter energy.")
    run_rd_every: int = Field(100, description="Number of steps to run the reaction diffusion simulation.")
    cell_types_patterns: List[Dict] = Field([], description="List of dictionaries with the parameters for the reaction diffusion simulation for each cell type. (RDPatternLibrary)")
    class Config:
        arbitrary_types_allowed = True
