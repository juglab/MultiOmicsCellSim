import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pydantic import BaseModel, Field
from typing import Literal, Union
from IPython.display import HTML

import concurrent.futures

import tqdm

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


class ReactionDiffusion():
    """
     Implements a reaction-diffusion model using pytorch
    """

    def __init__(self, config: ReactionDiffusionConfig):
        self.c = config

    def _clamp(x: torch.Tensor):
        """
        Clamp tensor values to [0, 1]
        """
        return torch.clamp(x, 0.0, 1.0)

    def init_simulation(self, type: Union[Literal["empty", "random_pixels", "square"], None]=None):
        """
        Defines a initial configuration for the grid based on the selected type
        or the provided initial configuration.

            Returns:
                A, B: torch.Tensor: Initial configuration of the grid                        
        """

        if self.c.initial_configuration is not None:
            A = self.c.initial_configuration
            B = 1.0 - A
        else:
            type = self.c.initial_configuration_type

            if type is None:
                raise ValueError("No initial configuration provided and no type selected. Please provide an initial configuration or select a type of initial configuration to generate.")

            A = torch.ones(size=[self.c.size, self.c.size], dtype=torch.float)
            B = torch.zeros(size=[self.size, self.c.size], dtype=torch.float)

            if type == "empty":
                pass

            elif type == "random_pixels":
                initial_pixels = int(self.c.size * self.c.size * self.c.initial_pixels_perc)
                rnd_rows = torch.randint(0, self.c.size, (initial_pixels,))
                rnd_cols = torch.randint(0, self.c.size, (initial_pixels,))
                A[rnd_rows, rnd_cols] = 0.0

                B = 1.0 - A

            elif type == "square":
                # Define a square in the center of the grid with size initial_square_size_perc
                square_size = int(self.c.size * self.c.initial_square_size_perc)
                start = (self.c.size - square_size) // 2
                end = start + square_size
                A[start:end, start:end] = 0.0
                B = 1.0 - A
                return A, B
        
        # Setup feed and kill masks
        self.f_grid = self.c.f * (self.c.f_mask if self.c.f_mask is not None else torch.ones(size=[self.c.size, self.c.size], dtype=torch.float))
        self.k_grid = self.c.k * (self.c.k_mask if self.c.k_mask is not None else torch.ones(size=[self.c.size, self.c.size], dtype=torch.float))

        return A, B
        
    def _get_kernel(self):
        """
        Get the diffusion kernel
        """
        kernel = torch.tensor([[0.05, 0.2, 0.05], 
                               [0.2, -1,    0.2], 
                               [0.05, 0.2, 0.05]], dtype=torch.float)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel
    
    def _gray_scott(self, a, b):
        """
        Gets a gray-scott model update for the a and b chemicals
        """
        kernel = self._get_kernel()
        A_ext = a.unsqueeze(0).unsqueeze(0)
        B_ext = b.unsqueeze(0).unsqueeze(0)
        
        # Diffusion terms using convolution
        A_diffusion = F.conv2d(A_ext, kernel, padding="same").squeeze()
        B_diffusion = F.conv2d(B_ext, kernel, padding="same").squeeze()

        # Reaction terms
        reaction = a * (b**2)
        
        # Update equations
        A_update = (self.c.d_A * A_diffusion - reaction + self.f_grid * (1 - a))*self.c.delta_t
        B_update = (self.c.d_B * B_diffusion + reaction - (self.f_grid + self.k_grid) * b)*self.c.delta_t

        A = a + A_update
        B = b + B_update

        return A, B
         
    def _stop_condition(self, a, b):
        """
        Check if the simulation should stop
        """
        return (a.mean().item() < self.c.convergence_threshold) and (b.mean().item() < self.c.convergence_threshold)

    def compute_model(self, a, b):
        if self.c.model == "Gray-Scott":
            return self._gray_scott(a, b)
    
    def yield_step(self, a, b):
        """
        Yield the next step of the model
        """
        for t in range(self.c.steps):
            # print(f"Step {t}")
            A, B = self.compute_model(a, b)
            yield A, B
            if self._stop_condition(A, B):
                break
        return A, B

    def run_until_convergence(self, cache_steps: bool=False, plot_every:int =None):
        """
        Run the model until convergence
        """
        steps = [self.init_simulation()]
        a, b = steps[0]

        for t in tqdm.tqdm(range(self.c.steps)):
            a, b = self.compute_model(a, b)
            if cache_steps:
                steps.append((a, b))
            if plot_every:
                if len(steps) % plot_every == 0:
                    self.plot(b)
            if self._stop_condition(a, b):
                break
                    
        if cache_steps:
            return steps
        else:
            return a, b

    def plot(self, x: torch.Tensor):
        """
        Plot the image
        """
        fig, ax = plt.subplots()
        ax = [ax]
        plt.imshow(x, cmap="viridis")
        ax[0].axis("off")
        ax[0].set_aspect("equal")
        ax[0].set_title(f"Reaction-Diffusion. Max value: {x.max().item():.2f}, Min value: {x.min().item():.2f}")
        plt.show()

    def show_animation(self, cached_steps: list, interval: int=50, show: Literal["A", "B"]="B") -> HTML:
        """
            Given a list of cached steps, show an animation of the simulation

        """
        fig, ax = plt.subplots()
        ax = [ax]
        ims = []

        def process_step(step):
            a, b = step
            to_show = a if show == "A" else b
            im = ax[0].imshow(to_show, cmap="viridis")
            ax[0].axis("off")
            ax[0].set_aspect("equal")
            ax[0].set_title(f"Reaction-Diffusion. Max value: {to_show.max().item():.2f}, Min value: {to_show.min().item():.2f}")
            return [im]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            ims = list(tqdm.tqdm(executor.map(process_step, cached_steps[::interval]), total=len(cached_steps[::interval])))
        return animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
         