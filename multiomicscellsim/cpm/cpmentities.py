from pydantic import BaseModel, Field, model_validator, computed_field
from functools import cached_property
from typing import Optional, List, Literal, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from .constraints import HamiltonianConstraint, AdhesionConstraint, VolumeConstraint



class CPMCellType(BaseModel):
    id: int = 0
    name: Optional[str] = Field("", description="Human Readable name of the cell type")
    # Constraint-specific parameters
    adhesion_energy: List[float] = Field([], description="List of Adhesion Energy between cell types. Must have an entry for each cell type in the simulation. Self-adhesion is ignored. First position (0) represents adhesion with the background / medium")
    preferred_volume: int = Field(9, description="Preferred volume for this cell type. Used with VolumeConstraint.")
    preferred_perimeter: int = Field(8, description="Preferred perimeter for this cell type. Used with PerimeterConstraint.")

class CPMCell(BaseModel):
    """
        Represents a single cell in the simulation. 
        It is used to keep track of the parameters that are specific to the single cell, such as type and current volume.
    """
    id: int = 0
    cell_type: CPMCellType
    volume: int
    _neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def set_neighbors(self, neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]]):
        self._neighbors = neighbors

    def gain_volume(self, gain: int = 1):
        self.volume += gain
        
    def update_perimeter(self, change: int):
        self.perimeter += change

    @computed_field
    @property
    def perimeter(self) -> int:
        return np.sum([len(nb) for k, nb in self._neighbors.items()])
    
    def add_neighbor_to(self, coords, neighbor):
        r, c = coords
        if (r, c) not in self._neighbors.keys():
            self._neighbors[(r, c)] = []
        if neighbor not in self._neighbors[(r, c)]:
            self._neighbors[(r, c)].append(tuple(neighbor))
    
    def remove_neighbor_from(self, coords, neighbor, all=False):
        r, c = coords
        if all:
            del self._neighbors[(r,c)]
        else:
            if (r, c) in self._neighbors:
                try:
                    self._neighbors[(r, c)].remove(tuple(neighbor))
                    # Clean up if no neighbors remain at (r, c)
                    if not self._neighbors[(r, c)]:
                        del self._neighbors[(r, c)]
                except ValueError:
                    # Neighbor was not in the list
                    pass

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
    temperature: float = Field(1.0, description="Temperature for the simulation. The higher the temperature, the more energetically unfavourable copy attempts will succeed.")
    neighborhood: Literal["moore", "von_neumann"] = Field("moore", description="Which kind of neighborhood to check.")
    constraints: List[HamiltonianConstraint] = Field([AdhesionConstraint(), VolumeConstraint()], description="List of HamiltonianConstraints to enable simulation constraint to influence cell behaviour.")
    cell_types: List[CPMCellType] = Field([], description="List of CPMCellTypes. Must be provided ordered by ID, starting from 1 and omitting the ID=0 which is assumed to be the background.")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def initialize_lattice(cls, values):
        size = values.get("size", 64)
        if "grid" not in values:
            values["grid"] = np.zeros((size, size, 3), dtype=np.int32)
        return values
    
    @computed_field
    @cached_property
    def _cells(self) -> List[CPMCell]:
        """
            Initializes the list of cells in the simulation.
            It contains reference to the current value for each property and its cell type.
            ID = 0 is a special id and it represents the background (stroma)
        """
        cells = []

        cell_ids = self.grid[..., 0]
        for cell_id in np.unique(cell_ids):
            # get one random (first) coordinate belonging to a cell with a give cell_id 
            cell_mask = self.mask_cell_id(cell_id=cell_id)
            rc = list(zip(*np.where(cell_mask)))[0]
            cell_type_id = self.grid[rc[0], rc[1], 1]
            cell_type = self.cell_types[cell_type_id]
            cell = CPMCell(
                id=cell_id,
                cell_type=cell_type,
                volume=np.sum(cell_mask)
            )
            cells.append(cell)
        return cells


    @cached_property
    def neighbors(self) -> List[List[Tuple[int, int]]]:
        """Precompute neighbors for each pixel."""
        print("Precomputing Neighbors...")

        if self.neighborhood == "moore":
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1), (1, 0), (1, 1)]
        elif self.neighborhood == "von_neumann":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise NotImplementedError(f"{self.neighborhood} not implemented.")

        neighbors_grid = []

        # Iterate over rows and columns
        for row in range(self.size):  # row represents the vertical index (height)
            row_neighbors = []
            for col in range(self.size):  # col represents the horizontal index (width)
                # Compute neighbors for the pixel at (row, col)
                neighbors = [(row + d_row, col + d_col) for d_row, d_col in offsets
                            if 0 <= row + d_row < self.size 
                            and 0 <= col + d_col < self.size]

                row_neighbors.append(neighbors)
            neighbors_grid.append(row_neighbors)

        return neighbors_grid

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
        for cell_id in range(n):  # Actual cell id is generated by the draw function
            # Randomly select a position in the grid until an empty one is found
            succeeded = False
            while not succeeded:
                r, c = np.random.randint(0, self.size), np.random.randint(0, self.size)
                succeeded = self.draw_cell_at([r, c], cell_type=np.random.choice(cell_types))
    
    
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
            r, c = valid_coords[np.random.choice(len(valid_coords))]
        else:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
        return (r, c), self.grid[r, c]

    def moore_neighborhood(self, r: int, c: int) -> List[Tuple[int, int]]:
            """
                Generates the Moore neighborhood (8 neighbors) for a given pixel at (r, c).
                Ensures neighbors are within grid bounds.

                Parameters
                ----------
                r, c: int - Coordinates of the source pixel

                Returns
                -------
                neighbors: List[Tuple[int, int]] - List of coordinates for the Moore neighborhood
            """
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the source pixel itself
                    nr, nc = r + dr, c + dc
                    # Check boundaries to avoid edge errors
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        neighbors.append((nr, nc))
            return neighbors

    def _compute_cell_neighbors(self, cell_id, z_layer=0):
        """
            Compute cell neighbors, considering only those that have a different value in the z_layer.
            
            Returns
            -------
            Dict[(int, int)]: List[(int, int)]
        """
        cell_mask = self.mask_cell_id(cell_id=cell_id)
        neighbors = dict()
        for r, c in zip(*np.where(cell_mask)):
            source_property = self.get_pixel([r, c])[z_layer]
            nbs = [nb for nb in self.neighbors[r][c] if (self.get_pixel(nb)[z_layer] != source_property)]
            if len(nbs) > 0:
                neighbors[(r, c)] = nbs
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
        
        for r in range(height):
            for c in range(width):
                cell_property = self.grid[r, c, z_layer]
                
                # Check neighbors
                for nr, nc in self.moore_neighborhood(r, c):
                    neighbor_cell_property = self.grid[nr, nc, z_layer]
                    
                    # If there's a neighboring cell with a different property, mark this pixel in the mask
                    if cell_property != neighbor_cell_property:
                        mask[r, c] = True
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
        r, c = source_coords
        if neighborhood == "moore":
            neighbors = self.moore_neighborhood(r, c)
        else:
            raise NotImplementedError("Unsupported neighborhood type. Only 'moore' is implemented.")
        
        if not neighbors:
            raise ValueError("No neighbors found, check lattice boundaries or neighborhood size.")
        
        # Pick a random neighbor
        neighbor_coords = neighbors[np.random.randint(len(neighbors))]
        return neighbor_coords, self.grid[neighbor_coords[0], neighbor_coords[1]]

    def draw_cell_at(self, pos: List[int], cell_type: int, size=3) -> bool:
        """
            Draws a cell at the given position if all pixels are empty.
            Updates the volume accordingly, only counting pixels inside the viewport.
            Cell ID is automatically computed by extending the current cell list.

            Returns
            -------
            Whether the drawing succeeded
        """
        cell_id = len(self._cells)  # ID 0 is the background, will be computed upon first access

        half_size = size // 2
        start_col = pos[1] - half_size
        end_col = pos[1] + half_size - (1 if size % 2 == 0 else 0)
        start_row = pos[0] - half_size
        end_row = pos[0] + half_size - (1 if size % 2 == 0 else 0)
        value = [cell_id, cell_type, 1]

        # Ensure boundaries are within grid limits
        start_col = max(start_col, 0)
        start_row = max(start_row, 0)
        end_col = min(end_col, self.size - 1)
        end_row = min(end_row, self.size - 1)

        # Check if the region is empty
        if (self.grid[start_row:end_row + 1, start_col:end_col + 1, 0] == 0).all():
            # Draw the cell by setting values in the specified region
            self.grid[start_row:end_row + 1, start_col:end_col + 1, :] = value

            # Calculate the area and update the volume for the cell
            drawn_area = (end_col - start_col + 1) * (end_row - start_row + 1)
            
            # Since we are directly accessing the grid, update the volume for the background
            self._cells[0].gain_volume(-drawn_area)

            new_cell = CPMCell(id=cell_id, 
                            cell_type=self.cell_types[cell_type], 
                            volume=drawn_area)
            
            # This has to be called AFTER drawing the cell!
            new_cell.set_neighbors(self._compute_cell_neighbors(cell_id, z_layer=0))
            
            self._cells.append(new_cell)

            return True
        return False

    def set_pixel(self, coords: List[int], value: np.ndarray):
        """
            Set a pixel to the target value, keeping cell volumes updated
        """
        
        # set pixel
        self.grid[coords[0], coords[1]] = value

    def copy_pixel(self, source, target):
        """
            Copy a pixel (and their other dimensions) from source coordinates to target coordinates and keeps all the 
            cell features updated.
        """

        new_value = self.get_pixel(source)
        new_cell_id, new_cell_type, new_organelles = new_value
        source_cell = self.get_cell(source)
        target_cell = self.get_cell(target)

        # update volumes
        source_cell.gain_volume(-1)
        target_cell.gain_volume(1)

        # update neighbors and perimeters
        new_nbs = [] # new row in neighbors table for the source cell
        for nb in self.neighbors[target[0]][target[1]]:
            nb_cell_id = self.get_cell_id(nb)
            if nb_cell_id == new_cell_id:
                if source_cell.id != 0:
                    source_cell.remove_neighbor_from(coords=nb, neighbor=target)
            else:
                new_nbs.append(nb)
                if nb_cell_id != 0 and target_cell.id !=0:
                    target_cell.add_neighbor_to(coords=nb, neighbor=target)

        # Adding all the neighbors belonging to the target pixel to the source cell
        for nb in new_nbs:
            if source_cell.id != 0:
                source_cell.add_neighbor_to(coords=target, neighbor=nb)
        # Removing all the neighbors belonging to the target pixel from the target cell
        if target_cell.id != 0:
            target_cell.remove_neighbor_from(target, neighbor=None, all=True)

        self.set_pixel(target, new_value)

    def get_pixel(self, coords: List[int]):
        return self.grid[coords[0], coords[1]]

    def get_cell(self, coords: List[int]) -> CPMCell:
        return self._cells[self.get_cell_id(coords=coords)]

    def get_cell_id(self, coords: List[int]) -> int:
        return self.get_pixel(coords=coords)[0]
    
    def get_cell_type(self, coords: List[int]) -> int:
        return self.get_pixel(coords=coords)[1]

    def mask_cell_id(self, cell_id) -> np.ndarray:
        """
            Returns a binary mask containing only a given cell.
        """
        return self.grid[..., 0] == cell_id

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