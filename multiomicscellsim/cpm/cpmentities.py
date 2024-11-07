from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
from .constraints import HamiltonianConstraint, AdhesionConstraint, VolumeConstraint



class CPMCellType(BaseModel):
    id: int = 0
    name: Optional[str] = Field("", description="Human Readable name of the cell type")
    # Constraint-specific parameters
    adhesion_energy: List[float] = Field([], description="List of Adhesion Energy between cell types. Must have an entry for each cell type in the simulation. Self-adhesion is ignored. First position (0) represents adhesion with the background / medium")
    preferred_volume: int = Field(9, description="Preferred volume for this cell type. Used with VolumeConstraint.")


class CPMGrid(BaseModel):
    """
        Represents a lattice in which the cells live and all the related simulation parameters.
        
        The grid is encoded as a 3-channel image, encoded as following:
        0: Cell ID
        1: Cell Type ID
        2: Subcellular Entities (e.g. 0 is Nothing, 1 is Cytoplasm, 2 is the Nucleus)

        This class includes all the parameters that are specific of a particular grid and those that are general to constraints (such as the weighting factor for VolumeConstraint).
    """

    size: int
    grid: np.ndarray
    dimensions: int = 3
    temperature: float = Field(1.0, description="Temperature for the simulation. The higher the temperature, the more energetically unfavourable copy attempts will succeed.")
    neighborhood: Literal["moore"] = Field("moore", description="Which kind of neighborhood to check.")
    constraints: List[HamiltonianConstraint] = Field([AdhesionConstraint(), VolumeConstraint()], description="List of HamiltonianConstraints to enable simulation constraint to influence cell behaviour.")
    cell_types: List[CPMCellType] = Field([], description="List of CPMCellTypes. Must be provided ordered by ID, starting from 1 and omitting the ID=0 which is assumed to be the background.")
    

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def initialize_lattice(cls, values):
        size = values.get("size", 64)
        dimensions = values.get("dimensions", 3)
        if "grid" not in values:
            values["grid"] = np.zeros((size, size, dimensions), dtype=np.int32)
        return values

    def spawn_random_cell(self, n: int):
        """
            Spawn `n` random cells in the grid, assigning unique cell IDs.
            
            Parameters
            ----------
            n : int
                The number of cells to spawn.
        """
        # TODO: Also consider the third layer
        # Iterate over each cell to be spawned
        cell_types = [ct.id for ct in self.cell_types if ct.id != 0]
        for cell_id in range(1, n + 1):  # Start cell IDs from 1 since 0 is the background
            # Randomly select a position in the grid until an empty one is found
            succeeded = False
            while not succeeded:
                x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
                succeeded = self.draw_cell_at([x, y], cell_id=cell_id, cell_type=np.random.choice(cell_types))
    
    
    def get_random_pixel(self, mask=None) -> Tuple[Tuple[int, int], np.ndarray]:
        """
            Returns a random grid pixel.

            Parameters
            -------
            mask: ndarray, pick only from pixels that are inside this mask. If None, can pick any pixel.

            Returns
            -------
            coords: Tuple[int, int] - The coordinates of the sampled pixel
            value: np.ndarray - the slice of the grid corresponding to coords
        """
        if mask is not None:
            valid_coords = list(zip(*np.where(mask)))
            x, y = valid_coords[np.random.choice(len(valid_coords))]
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
        return (x, y), self.grid[x, y]

    def moore_neighborhood(self, x: int, y: int) -> List[Tuple[int, int]]:
            """
                Generates the Moore neighborhood (8 neighbors) for a given pixel at (x, y).
                Ensures neighbors are within grid bounds.

                Parameters
                ----------
                x, y: int - Coordinates of the source pixel

                Returns
                -------
                neighbors: List[Tuple[int, int]] - List of coordinates for the Moore neighborhood
            """
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the source pixel itself
                    nx, ny = x + dx, y + dy
                    # Check boundaries to avoid edge errors
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        neighbors.append((nx, ny))
            return neighbors


    def get_frontier_mask(self, z_layer: int=0) -> np.ndarray:
        """
        Creates a mask where pixels are marked as True if they have at least one neighbor with a different property (e.g., cell type or cell id).
        
        Parameters
        -------
        z_value: which layer to use

        Returns
        -------
        mask: np.ndarray - A 2D boolean array of the same width and height as the grid
        """
        height, width = self.grid.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        
        for x in range(width):
            for y in range(height):
                cell_property = self.grid[x, y, z_layer]  # Get the cell property at (x, y)
                
                # Check neighbors
                for nx, ny in self.moore_neighborhood(x, y):
                    neighbor_cell_property = self.grid[nx, ny, z_layer]
                    
                    # If there's a neighboring cell with a different property, mark this pixel in the mask
                    if cell_property != neighbor_cell_property:
                        mask[x, y] = True
                        break  # No need to check further neighbors
        
        return mask

    def get_random_neighbour(self, source_coords: Tuple[int, int], neighborhood="moore") -> Tuple[Tuple[int, int], np.ndarray]:
        """
            Returns a random neighbor of the source pixel based on the specified neighborhood.

            Parameters
            ----------
            source_coords: Tuple[int, int] - Coordinates of the source pixel
            neighborhood: str - The type of neighborhood ("moore" for 8 neighbors)

            Returns
            -------
            neighbor_coords: Tuple[int, int] - Coordinates of the randomly chosen neighbor
            neighbor_value: np.ndarray - The slice of the grid corresponding to the neighbor
        """
        x, y = source_coords
        if neighborhood == "moore":
            neighbors = self.moore_neighborhood(x, y)
        else:
            raise NotImplementedError("Unsupported neighborhood type. Only 'moore' is implemented.")
        
        if not neighbors:
            raise ValueError("No neighbors found, check lattice boundaries or neighborhood size.")
        
        # Pick a random neighbor
        neighbor_coords = neighbors[np.random.randint(len(neighbors))]
        return neighbor_coords, self.grid[neighbor_coords[0], neighbor_coords[1]]

    def draw_cell_at(self, pos: List[int], cell_id: int, cell_type:int, size=3) -> bool:
        """
            Draws a cell at the given position if all pixels are empty.

            Returns
            -------
            Whether the drawing succeeded
        """
        start_x, end_x = pos[0] - size//2, pos[0] + size//2
        start_y, end_y = pos[1] - size//2, pos[1] + size//2
        value = [cell_id, cell_type, 1]
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        end_x = min(end_x, self.size)
        end_y = min(end_y, self.size)
        if (self.grid[start_x:end_x+1, start_y:end_y+1, 0] == 0).all():
            self.grid[start_x:end_x+1, start_y:end_y+1, :] = value
            return True
        return False

    def set_pixel(self, coords: List[int], value: np.ndarray):
        self.grid[coords[0], coords[1]] = value

    def get_pixel(self, coords: List[int]):
        return self.grid[coords[0], coords[1]]

    def get_cell_id(self, coords: List[int]) -> int:
        return self.get_pixel(coords=coords)[0]
    
    def get_cell_type(self, coords: List[int]) -> int:
        return self.get_pixel(coords=coords)[1]

    def mask_cell_id(self, cell_id) -> np.ndarray:
        """
            Returns a binary mask containing only a given cell.
        """
        return self.grid[..., 0] == cell_id

    def copy_pixel(self, source, target):
        """
            Copy a pixel (and their other dimensions) from source coordinates to target coordinates.
        """
        #print(f"Copy {source} to {target}")
        self.grid[target[0], target[1]] = self.grid[source[0], source[1]]

    def render(self, plot_layer=1):
        """
            Renders the current state of the simulation grid.
            Each cell ID will be represented by a different color.
        """
        # Extract the cell ID layer (first layer) for rendering
        cell_id_layer = self.grid[:, :, plot_layer]

        # Define a color map; 'n' colors based on unique cell IDs
        unique_ids = np.unique(cell_id_layer)
        color_map = plt.cm.get_cmap("viridis", len(unique_ids))

        # Map each cell ID to a unique color index
        color_indices = np.searchsorted(unique_ids, cell_id_layer)

        # Plot the grid using the color indices
        plt.imshow(color_indices, cmap=color_map, origin="upper")
        plt.colorbar(label="Cell ID" if plot_layer == 0 else "Cell Type")
        plt.title("Cellular Potts Model Simulation")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()