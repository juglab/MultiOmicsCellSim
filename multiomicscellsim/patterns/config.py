from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, Union, Literal
import torch
from functools import cached_property
from pathlib import Path
import json
import random


class RDPatternConfig(BaseModel):
    pattern_name: str = Field(..., description="Name of the pattern")
    d_a: float = Field(0.2097, description="Diffusion rate for A")
    d_b: float = Field(0.1050, description="Diffusion rate for B")
    f: float = Field(0.0540, description="Feed rate to use. If a mask is provided, this value is multiplied by the mask.")
    k: float = Field(0.0620, description="Kill rate. If a mask is provided, this value is multiplied by the mask.")

    def add_noise(self, to: List, var: float = 0.01):
        new = self.model_copy().model_dump()
        for param in to:
            new[param] = random.gauss(mu=new[param], sigma=var)
        return RDPatternConfig(**new)

class RDPatternLibrary:
    """
    Library of reaction-diffusion patterns
    Presets based on Robert Munafo's (mrob's) WebGL
    Gray-Scott Explorer:  https://mrob.com/pub/comp/xmorphia/ogl/index.html
    and Jason Webb: https://github.com/jasonwebb/reaction-diffusion-playground/
    """
    patterns = []

    @staticmethod
    def load_patterns_from_file(file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            RDPatternLibrary.patterns = [RDPatternConfig(**pattern) for pattern in data["patterns"]]

    @staticmethod
    def get_pattern_by_name(pattern_name: str) -> RDPatternConfig:
        for pattern in RDPatternLibrary.patterns:
            if pattern.pattern_name == pattern_name:
                return pattern
        raise ValueError(f"Pattern {pattern_name} not found in patterns. Available patterns: {RDPatternLibrary.patterns}")
    
    @staticmethod
    def get_pattern_names() -> list:
        return [pattern.pattern_name for pattern in RDPatternLibrary.patterns]
    
    @staticmethod
    def get_pattern_by_index(index: int) -> RDPatternConfig:
        return RDPatternLibrary.patterns[index]


# Load patterns from the file
RDPatternLibrary.load_patterns_from_file(Path(__file__).parent / "patterns.json")

class ReactionDiffusionConfig(BaseModel):
    size: int = Field(256, description="Size of the grid")
    steps: int = Field(5000, description="Number of steps to run the simulation")
    initial_configuration: Union[None, torch.Tensor] = Field(None, description="Initial configuration of the grid. Can be used to set a custom initial configuration.")
    initial_configuration_type: Union[Literal["empty", "random_pixels", "square"], None] = Field(None, description="Type of initial configuration to generate. Required if no initial_configuration is provided.")
    initial_pixels_perc: float = Field(0.05, description="Percentage of initial pixels with A=0 to use if no initial configuration is provided and the random_pixels option is selected in generate_initial_configuration")
    initial_square_size_perc: float = Field(0.1, description="Size of the square to use if no initial configuration is provided and the square option is selected in generate_initial_configuration")

    mask_output: torch.Tensor = Field(None, description="Mask to apply to the output of the simulation.")

    delta_t: float = Field(1.0, description="Time step")
    plot_every: int = Field(50, description="Plot every N steps")
    convergence_threshold: float = Field(1e-5, description="Convergence threshold")
    
    model: Literal["Gray-Scott"] = Field("Gray-Scott", description="Model to use")

    d_A: float = Field(0.2097, description="Diffusion rate for A")
    d_B: float = Field(0.1050, description="Diffusion rate for B")
    f: float = Field(0.0540, description="Feed rate to use. If a mask is provided, this value is multiplied by the mask.")
    k: float = Field(0.0620, description="Kill rate. If a mask is provided, this value is multiplied by the mask.")
    f_mask: Union[None, torch.Tensor] = Field(None, description="Feed rate mask. Gets multiplied by the feed rate scalar.")
    k_mask: Union[None, torch.Tensor] = Field(None, description="Kill rate mask. Gets multiplied by the kill rate scalar.")

    # Allow arbitrary types to be used in the model (e.g. torch.Tensor)
    class Config:
        arbitrary_types_allowed = True