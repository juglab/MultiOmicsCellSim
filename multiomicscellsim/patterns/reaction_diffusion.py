import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Literal, Union
from IPython.display import HTML

import concurrent.futures

from tqdm import tqdm

from .config import ReactionDiffusionConfig


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
            B = torch.zeros(size=[self.c.size, self.c.size], dtype=torch.float)

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
    
    def run_on_cpm_grid(self, grid: torch.Tensor, steps: int, f: float, k: float, d_a: float, d_b: float):
        """
           Run simulation on a CPM grid for the given number of steps

              Args:
                grid (torch.Tensor): Grid to run the simulation on. Should be [2, h, w], where the channels are: [A, B]
                steps (int): Number of steps to run the simulation
                f (float): Feed rate
                k (float): Kill rate
                d_a (float): Diffusion rate for A
                d_b (float): Diffusion rate for B
        """

        a = grid[0]
        b = grid[1]

        for t in tqdm(range(steps), leave=False):
            a, b = self._gray_scott(a, b, f, k, d_a, d_b)
        
        grid[0] = a
        grid[1] = b
        return grid
        
        

    def _gray_scott(self, a: torch.Tensor, b: torch.Tensor, f_grid: torch.Tensor, k_grid: torch.Tensor, d_a: float, d_b: float, delta_t: float=1.0):
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
        A_update = (d_a * A_diffusion - reaction + f_grid * (1 - a))*delta_t
        B_update = (d_b * B_diffusion + reaction - (f_grid + k_grid) * b)*delta_t

        A = a + A_update
        B = b + B_update

        return A, B
         
    def _stop_condition(self, a, b):
        """
        Check if the simulation should stop
        """
        return (a.mean().item() < self.c.convergence_threshold) and (b.mean().item() < self.c.convergence_threshold)

    def compute_model_from_config(self, a, b):
        if self.c.model == "Gray-Scott":
            return self._gray_scott(a, b, self.f_grid, self.k_grid, self.c.d_A, self.c.d_B, self.c.delta_t)
    
    def yield_step_from_config(self, a, b):
        """
        Yield the next step of the model
        """
        for t in range(self.c.steps):
            # print(f"Step {t}")
            A, B = self.compute_model_from_config(a, b)
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

        for t in tqdm(range(self.c.steps)):
            a, b = self.compute_model_from_config(a, b)
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

    def show_animation(self, cached_steps: list, interval: int=50, show: Literal["A", "B"]="B", mask=None) -> HTML:
        """
            Given a list of cached steps, show an animation of the simulation

        """
        fig, ax = plt.subplots()
        ax = [ax]
        ims = []

        def process_step(step):
            a, b = step
            to_show = a if show == "A" else b
            if mask:
                to_show = to_show * mask
            im = ax[0].imshow(to_show, cmap="viridis")
            ax[0].axis("off")
            ax[0].set_aspect("equal")
            ax[0].set_title(f"Step {step} Max value: {to_show.max().item():.2f}, Min value: {to_show.min().item():.2f}")
            return [im]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            ims = list(tqdm(executor.map(process_step, cached_steps[::interval]), total=len(cached_steps[::interval])))
        return animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
         