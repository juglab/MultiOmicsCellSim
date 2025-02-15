import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from .config import TorchCPMConfig, TorchCPMCell
from .gridutils import vectorized_moore_neighborhood, get_differet_neigborhood_mask, get_frontiers, choose_random_neighbor, copy_source_to_neighbors, smallest_square_crop, get_different_neighbors, get_regular_shape_mask
from .constraints import TorchCPMAdhesionConstraint, TorchCPMVolumeConstraint, TorchCPMLocalPerimeterConstraint
from multiomicscellsim.patterns.reaction_diffusion import ReactionDiffusionConfig, ReactionDiffusion
from typing import Literal, Union, Dict

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
    # Dictionary containing the cell properties (f, k, d_a, d_b) for each cell_id
    cells: Dict[int, TorchCPMCell]

    def __init__(self, config: TorchCPMConfig):
        self.config = config
    
        self.grid = (torch.zeros((2, config.size, config.size), dtype=torch.int) - 1).to(self.config.device)
        self.subgrid = torch.zeros((2, config.size, config.size), dtype=torch.float).to(self.config.device)

        self.constraints = {"adhesion": TorchCPMAdhesionConstraint(config), 
                            "volume": TorchCPMVolumeConstraint(config),
                            "perimeter": TorchCPMLocalPerimeterConstraint(config)}
        
        # Get a reaction diffusion simulation with standard parameters
        self.rd = ReactionDiffusion(ReactionDiffusionConfig())
        self.cells = {}

    def draw_cell(self, xc: int, yc: int, radius: float, edges: int, orientation:float, cell_type_id: int, subcell_init_noise: Literal["binomial"] = None) -> Union[TorchCPMCell, None]:
        """
            Draw a cell as a regular polygon (or circle if edges == 0) on the grid.
            Returns the cell_id of the new cell on success (there is free space to draw the cell), 0 otherwise.

            Args:
                xc (float): X-coordinate of the center.
                yc (float): Y-coordinate of the center.
                radius (float): Radius of the circle or circumscribed circle of the polygon.
                edges (int): Number of edges for the polygon. If 0, draw a circle.
                image_size (int): Size of the square image (image_size x image_size).
                orientation (float): Orientation of the first edge or vertex in radians.
            
            Return:
                TorchCPMCell: The cell object created, or None if the cell could not be drawn (overlaps with existing cells).

        """
        if cell_type_id not in [cell_type.id for cell_type in self.config.cell_types]:
            raise ValueError(f"Cell type {cell_type_id} not defined in the configuration.")
        cell_type = next(cell_type for cell_type in self.config.cell_types if cell_type.id == cell_type_id)

        grid_size = self.config.size

        if xc < 0 or xc >= grid_size or yc < 0 or yc >= grid_size:
            return None
        
        # Get a mask containing the new cell
        new_cell_mask = get_regular_shape_mask(xc, yc, radius, edges, grid_size, orientation)
        
        # Checking overlaps between existing cells and the new cell
        if ((self.grid[0].cpu() > 0) & new_cell_mask).any():
            return None
        
        # Sampling cell parameters

        # Add a new cell id (Avoiding 0, which is reserved)
        cell_id = int(max(self.grid[0].max().item() + 1, 1))

        # Sampling reaction diffusion parameters from the cell type and creating the cell
        new_cell = TorchCPMCell(id=cell_id,
                                cell_type=cell_type,
                                f=float(cell_type.subcell_f_std * torch.randn(1).item() + cell_type.subcellular_pattern.f),
                                k=float(cell_type.subcell_k_std * torch.randn(1).item() + cell_type.subcellular_pattern.k),
                                d_a=float(cell_type.subcell_da_std * torch.randn(1).item() + cell_type.subcellular_pattern.d_a),
                                d_b=float(cell_type.subcell_db_std * torch.randn(1).item() + cell_type.subcellular_pattern.d_b),
                                )
        self.cells[cell_id] = new_cell

        # Updating the grid with the new cell
        self.grid[0][new_cell_mask > 0] = cell_id
        self.grid[1][new_cell_mask > 0] = cell_type_id

        # Set the initial state for subcellular layers
        # The cell starts full of A, while B fills the stroma
        if subcell_init_noise == "bernoulli":
            self.subgrid[0][new_cell_mask > 0] = (torch.rand((new_cell_mask > 0).sum()) > .5).float().to(self.config.device)
        else:
            self.subgrid[0][new_cell_mask > 0] = 1.0
        self.subgrid[1] = 1.0-self.subgrid[0]

        return self.cells[cell_id]


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
      

    def yield_step(self, yield_every: int = 1):
        """
            Yields results of the simulation.

            A simulation loop is composed of:
                0: 1 CPM Step
                1: rd_warmup_steps RD steps
                2: run_rd_every CPM steps
                3: rd_steps RD steps
                4: Every "yield_every" steps, the a result is yielded
                5: Loop to 2 until max_cpm_steps is reached.           
                
            Args:
                - yield_every: 
                    Yield a simulation result every n simulation steps. If 0, only returns the last step
                    The first and the last step are always yielded regardless of this parameter.
                    Default: 1

            Returns: 
                - A Generator of simulation results. 
                    Each simulation result is a Tuple of length 4:
                        - grid: [torch.Tensor] of shape [2, H, W] and dtype int.
                            grid[0] contains the cell ids (-1: background, > 0 cells)
                            grid[1] contains the cell type ids (-1: background, > 0 cell types)
                        - subgrid: [torch.Tensor] of shape [2, H, W] and dtype float.
                            subgrid[0] contains the component A of reaction diffusion
                            subgrid[1] contains the component B of reaction diffusion
                        - current cpm step (int)
                        - current rd step (int)

        """
        total_rd_steps = 0
        for cpm_step in tqdm(range(self.config.max_cpm_steps)):
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
            should_run_rd = (cpm_step % self.config.run_rd_every) == 0
            is_last_step = cpm_step == self.config.max_cpm_steps - 1
            should_yield = (cpm_step % yield_every) == 0 if yield_every > 0 else is_last_step
            if self.config.run_rd_every and (should_run_rd or is_last_step):
                rd_steps = self.config.rd_steps if cpm_step > 0 else self.config.rd_warmup_steps
                x_after, s_after = self.run_reaction_diffusion_on_cells(x_after, s_after, rd_steps)
                total_rd_steps += rd_steps
            if should_yield or is_last_step:
                yield x_after.cpu(), s_after.cpu(), cpm_step, total_rd_steps

            self.grid = x_after
            self.subgrid = s_after


    def run_reaction_diffusion_on_cells(self, grid_state: torch.Tensor, subgrid_state: torch.Tensor, steps: int):
        """
            Isolates each cell in the simulation and run the given amoutn of reaction diffusion steps on them.

            Args:
                grid_state (torch.Tensor): Grid state of the simulation. Shape: [2, H, W]. Channels: 0: Cell_ID, 1: Cell_Type
                subgrid_state (torch.Tensor): Subcellular grid state of the simulation. Shape: [2, H, W]. Channels: 0: Subcellular_A, 1: Subcellular_B
                steps (int): Number of reaction diffusion steps to run on each cell.
        """

        # Iterate over all cells in the grid (the ones that exist, self.cell might contain cells with no pixels)
        for cell_id in grid_state[0].unique():
            cell_id = int(cell_id.item())
            if cell_id <= 0:
                continue
            # Fetch the cell parameters
            rd_params = self.cells[cell_id]

            # Only pass to the reaction diffusion simulation the sub-square containing the cell
            crop_mask, (start_row, end_row, start_col, end_col) = smallest_square_crop(grid_state[0]==cell_id)

            # Mask the outer area of the cell to avoid including surrounding cells in the simulation
            # The content of A and B inside the cell (crop_mask==1.0) is always preserved
            # Outside the cell is always 0.0 for A and 1.0 for B
            
            cropped_subgrid = torch.stack([
                subgrid_state[0, start_row:end_row, start_col:end_col]*crop_mask,
                subgrid_state[1, start_row:end_row, start_col:end_col]*crop_mask + (1.0-crop_mask.float())
            ], dim=0)

            # TODO: Run this in batch for multiple cells
            # Run the reaction diffusion simulation
            rd_output = self.rd.run_on_cpm_grid(cropped_subgrid.clone(), 
                                        steps=steps, 
                                        f=rd_params.f, 
                                        k=rd_params.k, 
                                        d_a=rd_params.d_a, 
                                        d_b=rd_params.d_b)


            # Paste back the cell into the subcellular grid, preserving areas that are not part of the cell (if cell is round then the corners must keep the original value)
            updated_A = torch.where(crop_mask, rd_output[0], subgrid_state[0, start_row:end_row, start_col:end_col])
            updated_B = torch.where(crop_mask, rd_output[1], subgrid_state[1, start_row:end_row, start_col:end_col])
            subgrid_state[0, start_row:end_row, start_col:end_col] = updated_A
            subgrid_state[1, start_row:end_row, start_col:end_col] = updated_B
        return grid_state, subgrid_state
