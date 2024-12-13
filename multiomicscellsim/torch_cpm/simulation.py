import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from .config import TorchCPMConfig
from .gridutils import vectorized_moore_neighborhood, get_differet_neigborhood_mask, get_frontiers, choose_random_neighbor, copy_source_to_neighbors, smallest_square_crop, get_different_neighbors, copy_subgrid_to_neighbors
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

    # Grid of the simulation. Has dimensions [2, H, W]. Channels: 0: Cell_ID, 1: Cell_Type 
    grid: torch.Tensor
    # Subcellular grid of the simulation. Has dimensions [2, H, W]. Channels: 0: Subcellular_A, 1: Subcellular_B
    subgrid: torch.Tensor

    def __init__(self, config: TorchCPMConfig):
        self.config = config
    
        self.grid = torch.zeros((2, config.size, config.size), dtype=torch.int) - 1
        self.subgrid = torch.zeros((2, config.size, config.size), dtype=torch.float)

        self.constraints = {"adhesion": TorchCPMAdhesionConstraint(config), 
                            "volume": TorchCPMVolumeConstraint(config),
                            "perimeter": TorchCPMLocalPerimeterConstraint(config)}

    def draw_cell(self, x: int, y: int, cell_type: int, size: int = 11):
        """
            Draws a cell in the grid.
        """
        if cell_type not in [cell_type.id for cell_type in self.config.cell_types]:
            raise ValueError(f"Cell type {cell_type} not defined in the configuration.")
        
        # Add a new cell id (Avoiding 0, which is reserved)
        self.grid[0, x-size:x+size, y-size:y+size] = max(self.grid[0].max() + 1, 1)
        # Set the cell type
        self.grid[1, x-size:x+size, y-size:y+size] = cell_type
        # Set the subcellular pattern
        # This will be overwritten by the reaction diffusion simulation
        self.subgrid[1, x-size:x+size, y-size:y+size] = 1.0
        self.subgrid[0] = 1.0-self.subgrid[1]



    def plot_grid(self):
        """
            Plots the grid: Cell_ID, Cell_Type, Subcellular_A, Subcellular_B
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot Cell_ID
        ax = axs[0, 0]
        ax.imshow(self.grid[0].cpu().numpy(), cmap="tab20b", interpolation="none")
        ax.set_title("Cell_ID")
        
        # Plot Cell_Type
        ax = axs[0, 1]
        ax.imshow(self.grid[1].cpu().numpy(), cmap="tab20b", interpolation="none")
        ax.set_title("Cell_Type")
        
        # Plot Subcellular_A
        ax = axs[1, 0]
        ax.imshow(self.subgrid[0].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Subcellular_A")
        
        # Plot Subcellular_B
        ax = axs[1, 1]
        ax.imshow(self.subgrid[1].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Subcellular_B")
        
        plt.tight_layout()
        plt.show()
      

    def yield_step(self, max_steps: int):
        """
            Yields a step of the simulation.
        """
        

        # Get a reaction diffusion simulation with standard parameters
        rd = ReactionDiffusion(ReactionDiffusionConfig())

        for i in tqdm(range(max_steps)):
            x = self.grid
            s = self.subgrid

            # All neighbors indices (including thodse whithin the same object) - [8, h, w], val = -1 for background, >0 for the object id in each direction
            all_nbs = vectorized_moore_neighborhood(x)
            # All neighbors where neighbor != pixel (Binary Mask)
            diff_nbs_mask = get_differet_neigborhood_mask(x)
            # All neighbors where neighbor != pixel (val = -1 for background, 0 for same_value and >0 for the object id)
            diff_nbs = diff_nbs_mask * all_nbs    

            # Select source Pixels
            # Spatial Areas where the neighbors are different from the current pixel - [h, w]
            f = get_frontiers(x)
            # Source Pixels are the pixels that are selected to be copied onto one neighbor - [l, h, w]
            source_pixels = (torch.rand_like(f.float()) < self.config.frontier_probability) * f * x 

            # All neighbors (only from different areas), filtered by the source pixels - [8, h, w], Binary Mask
            source_diff_nbs_mask = diff_nbs_mask * (source_pixels[0].unsqueeze(0) != 0)
            # Choose a random neighbor for each source pixel with equal probability - [8, h, w], val = 1.0 for the chosen neighbor
            chosen_neighbors_mask = choose_random_neighbor(source_diff_nbs_mask)
            # Direction and value of the chosen neighbors - [8, h, w], val = -1 for background, 0 for no-choice and >0 for the object id
            chosen_neighbors = chosen_neighbors_mask * all_nbs

            # Energy calculation
            # Simulate a next state for each pixel in which the source pixels are copied to the chosen neighbors
            predicted_state = copy_source_to_neighbors(self.grid, source_pixels, chosen_neighbors)
            predicted_diff_neighbors = get_different_neighbors(predicted_state)
            delta_volume = self.constraints["volume"](x, diff_nbs, predicted_state, predicted_diff_neighbors)
            delta_adhesion = self.constraints["adhesion"](x, diff_nbs, predicted_state, predicted_diff_neighbors)
            delta_perimeter = self.constraints["perimeter"](x, diff_nbs, predicted_state, predicted_diff_neighbors)

            # Total Energy
            total_delta_energy =  delta_volume + delta_perimeter + delta_adhesion

            # Boltzmann distribution (only where the source pixels are)
            boltz_chance = (torch.rand_like(total_delta_energy) < torch.exp(-total_delta_energy/self.config.temperature))*(source_pixels[0]!=0)

            # We accept the copy if the energy is decreased or with a probability - [h, w], v: bool
            passed_pixel_mask = torch.logical_or(total_delta_energy < 0, boltz_chance)

            passed_neighbors = chosen_neighbors * passed_pixel_mask.unsqueeze(0)

            # Update the image by iterating over the directions and copying the source pixel value to the corresponding target pixel
            x_after = copy_source_to_neighbors(x, source_pixels, passed_neighbors)
            s_after = copy_source_to_neighbors(s, s, passed_neighbors, padding_value=0.0)
            
            # Run a reaction diffusion simulation on the cells
            if self.config.run_rd_every and (i % self.config.run_rd_every == 0 or i == max_steps - 1):
                

                for cell_type_id in x_after[1].int().unique():
                    # Get configuration for the cell type
                    if cell_type_id == 0:
                        continue
                    cell_type_config = self.config.cell_types[cell_type_id-1]
                    rd_params = cell_type_config.subcellular_pattern
                    if rd_params is None:
                        continue
                    
                    for cell_id in x_after[:, x_after[1]==cell_type_id].unique():
                        if cell_id <= 0:
                            continue

                        crop_mask, (start_row, end_row, start_col, end_col) = smallest_square_crop(x_after[0]==cell_id)

                        # TODO: Run this in batches
                        # Run the reaction diffusion simulation
                        rd_output = rd.run_on_cpm_grid(s_after[:, start_row:end_row, start_col:end_col].clone(), 
                                                  steps=self.config.rd_steps, 
                                                  f=rd_params.f, 
                                                  k=rd_params.k, 
                                                  d_a=rd_params.d_a, 
                                                  d_b=rd_params.d_b)
        

                        # Paste back the cell into the subcellular grid, preserving areas that are not part of the cell (if cell is round then the corners must keep the original value)
                        s_after[0, start_row:end_row, start_col:end_col] = torch.where(crop_mask, rd_output[0], s_after[0, start_row:end_row, start_col:end_col])
                        s_after[1, start_row:end_row, start_col:end_col] = torch.where(crop_mask, rd_output[1], s_after[1, start_row:end_row, start_col:end_col])
                       
            yield x_after, s_after
            self.grid = x_after
            self.subgrid = s_after


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


