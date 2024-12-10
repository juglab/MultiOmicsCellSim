import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from .config import TorchCPMConfig
from .gridutils import vectorized_moore_neighborhood, get_differet_neigborhood_mask, get_frontiers, choose_random_neighbor, copy_source_to_neighbors, smallest_square_crop
from .constraints import TorchCPMAdhesionConstraint, TorchCPMVolumeConstraint, TorchCPMLocalPerimeterConstraint
from multiomicscellsim.patterns.reaction_diffusion import ReactionDiffusionConfig, ReactionDiffusion


class TorchCPM():
    """
        Implements a Cellular Potts Model approximation using pytorch.
        This implementation differs from the original CPM since it tries to parallelize the metropolis algorithm
        to obtain better performances and easier integration with research code and programmatical definition of experiments.

        This comes at the cost of a slightly different implementation (and possibly behavior) of the algorithm
        due to the complexity in vectorizing all the operations.

        Differences:
         - The perimeter constrint is defined pixel-wise rather than cell-wise.
           This allows for an easier choice of parameters that still control the roughness of each cell.
         - Pixels are selected in group rather then one by one. This may introduce some bias in corner cases.    
    """

    def __init__(self, config: TorchCPMConfig, initial_state:torch.Tensor = None):
        self.config = config
        if initial_state is None:
            self.grid = torch.zeros((config.size, config.size), dtype=torch.int16) - 1
            self.subcellular_grid = torch.zeros((2, config.size, config.size), dtype=torch.float)
        else:
            self.grid = initial_state
        self.constraints = {"adhesion": TorchCPMAdhesionConstraint(config), 
                            "volume": TorchCPMVolumeConstraint(config),
                            "perimeter": TorchCPMLocalPerimeterConstraint(config)}

        # The grid indices cells by their id, this list maps the cell id to the cell type
        # (indices is cell id, value is cell type, 0 is unused (cell_id -1 is background, 0 is unselected))
        self.cell_types = [0]

    def draw_cell(self, x: int, y: int, cell_type: int, size: int = 11):
        """
            Draws a cell in the grid.
        """
        self.cell_types.append(cell_type)
        self.grid[x-size:x+size, y-size:y+size] = len(self.cell_types) - 1

    def plot_grid(self):
        """
            Plots the grid.
        """
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.grid)
        ax[0].set_title("Cellular Potts Model")
        ax[1].imshow(self.subcellular_grid[1])
        ax[1].set_title("Subcellular Grid")
        plt.show()
        


    def yield_step(self, max_steps: int):
        """
            Yields a step of the simulation.
        """
        
        for i in tqdm(range(max_steps)):
            x = self.grid
            # All neighbors indices (including thodse whithin the same object) - [8, h, w], val = -1 for background, >0 for the object id in each direction
            all_nbs = vectorized_moore_neighborhood(x)
            # All neighbors where neighbor != pixel (Binary Mask)
            diff_nbs_mask = get_differet_neigborhood_mask(x)
            # All neighbors where neighbor != pixel (val = -1 for background, 0 for same_value and >0 for the object id)
            diff_nbs = diff_nbs_mask * all_nbs    

            # Select source Pixels
            # Spatial Areas where the neighbors are different from the current pixel - [h, w]
            f = get_frontiers(x)
            # Source Pixels are the pixels that are selected to be copied onto one neighbor - [h, w]
            source_pixels = (torch.rand_like(f.float()) < self.config.frontier_probability) * f * x 

            # All neighbors (only from different areas), filtered by the source pixels - [8, h, w], Binary Mask
            source_diff_nbs_mask = diff_nbs_mask * (source_pixels.unsqueeze(0) != 0)
            # Choose a random neighbor for each source pixel with equal probability - [8, h, w], val = 1.0 for the chosen neighbor
            chosen_neighbors_mask = choose_random_neighbor(source_diff_nbs_mask)
            # Direction and value of the chosen neighbors - [8, h, w], val = -1 for background, 0 for no-choice and >0 for the object id
            chosen_neighbors = chosen_neighbors_mask * all_nbs

            # Energy calculation
            # Simulate a next state for each pixel in which the source pixels are copied to the chosen neighbors
            predicted_state, predicted_diff_neighbors = copy_source_to_neighbors(x, source_pixels, chosen_neighbors)
            delta_volume = self.constraints["volume"](x, diff_nbs, predicted_state, predicted_diff_neighbors)
            delta_adhesion = self.constraints["adhesion"](x, diff_nbs, predicted_state, predicted_diff_neighbors)
            delta_perimeter = self.constraints["perimeter"](x, diff_nbs, predicted_state, predicted_diff_neighbors)

            # Total Energy
            total_delta_energy =  delta_volume + delta_perimeter + delta_adhesion

            # Boltzmann distribution (only where the source pixels are)
            boltz_chance = (torch.rand_like(total_delta_energy) < torch.exp(-total_delta_energy/self.config.temperature))*(source_pixels!=0)

            # We accept the copy if the energy is decreased or with a probability - [h, w], v: bool
            passed_pixel_mask = torch.logical_or(total_delta_energy < 0, boltz_chance)

            ## Check this
            passed_neighbors = chosen_neighbors * passed_pixel_mask.unsqueeze(0)

            # Update the image by iterating over the directions and copying the source pixel value to the corresponding target pixel
            x_after, _ = copy_source_to_neighbors(x, source_pixels, passed_neighbors)
            
            # Run a reaction diffusion simulation on the cells
            if self.config.run_rd_every and (i % self.config.run_rd_every == 0 or i == max_steps - 1):
                self.run_reaction_diffusion_on_cells()
            yield x_after, self.subcellular_grid
            self.grid = x_after


    def run_reaction_diffusion_on_cells(self):
        """
            Run a reaction diffusion simulation on a cell.
        """
        # TODO: get a configuration based on cell_type. 
        # TODO: Implement a way to pass A and B to the simulation and do a certain number of steps

        for cell_id in self.grid.unique():
            if cell_id <= 0:
                continue
            crop, (start_row, end_row, start_col, end_col) = smallest_square_crop(self.grid==cell_id)
            rd_config = {
                        "size": crop.size(0),
                        "steps": 10000,
                        "initial_configuration_type": None,
                        "initial_configuration": crop.float(),
                        "initial_pixels_perc": 0.10,
                        "initial_square_size_perc": 0.1,
                        "delta_t": 1.0,
                        "plot_every": 50,
                        "convergence_threshold": 1e-5,
                        "model": "Gray-Scott",
                        "d_A": 0.2306,
                        "d_B": 0.1050
                    }
            # Apply the f, k parameters from the pattern
            rd_config.update(self.config.cell_types_patterns[cell_id])
            rd = ReactionDiffusion(ReactionDiffusionConfig(**rd_config))
            a, b = rd.run_until_convergence()
            # Paste back the cell into the subcellular grid, preserving areas that are not part of the cell (if cell is round then the corners must keep the original value)
            self.subcellular_grid[0, start_row:end_row, start_col:end_col] = torch.where(crop, a, self.subcellular_grid[0, start_row:end_row, start_col:end_col])
            self.subcellular_grid[1, start_row:end_row, start_col:end_col] = torch.where(crop, b, self.subcellular_grid[0, start_row:end_row, start_col:end_col])


