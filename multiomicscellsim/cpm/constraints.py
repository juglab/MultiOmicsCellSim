import numpy as np
from pydantic import BaseModel, Field
from typing import List


class HamiltonianConstraint(BaseModel):
    """
        Represents an Hamiltonian term that imposes a constraint on the simulation
    """


class AdhesionConstraint(HamiltonianConstraint):
    """
        Pixels of the same cell will be constrained to stick together
    """
    
    name: str = "AdhesionConstraint"

    def delta(self, source, target, grid: "CPMGrid"):
        """
            Calculates the change in Hamiltonian energy (ΔH) for the target pixel 
            if the source pixel value were copied there.
        """

        # Implementing it as in https://github.com/ingewortel/artistoo/blob/master/src/hamiltonian/Adhesion.js
        # simulating a target change in type

        # Initial adhesion energy at the target
        source_type = grid.get_cell_type(source)
        target_type = grid.get_cell_type(target)
        return self._h(target, source_type, grid=grid) - self._h(target, target_type, grid=grid)
               
    def _h(self, coords: List[int], cell_type: int, grid: "CPMGrid"):
        """
            Returns the Hamiltonian around a given cell by checking all the neighbors that have a different cell type.
        """
        # TODO: What about cell_id?
        return np.sum([nb for nb in grid.neighbors[coords[0]][coords[1]] if (grid.get_cell_type(nb) != cell_type)])

class VolumeConstraint(HamiltonianConstraint):
    name: str = "VolumeConstraint"
    lambda_volume: float = Field(1.0, description="Energy multiplier for the Volume Hamiltonian") 
    
    def delta(self, source: List[int], target: List[int], grid: "CPMGrid"):
        """
            Calculates the change in Hamiltonian energy (ΔH) for the target pixel 
            if the source pixel value were copied there.
        """
        source_cell = grid.get_cell(source)
        target_cell = grid.get_cell(target)

        # Delta for the source cell (energy after gain - current energy)
        delta_h_s = self._h(source_cell, gain=1) - self._h(source_cell, gain=0)
        # Delta for the target cell (energy after gain - current energy)
        delta_h_t = self._h(target_cell, gain=-1) - self._h(target_cell, gain=0)
        
        return delta_h_s + delta_h_t

    def _h(self, cell: "CPMCell",  gain: int):
        """
            Calculate the energy in case a given cell with id cell_id gains or loses a pixel.
            Adapted from:
            https://github.com/ingewortel/artistoo/blob/master/src/hamiltonian/VolumeConstraint.js
        """
        if cell.id == 0:
            # stroma is not involved in this constraint
            return 0
        
        volume_diff = cell.cell_type.preferred_volume - (cell.volume + gain)
        return self.lambda_volume * volume_diff * volume_diff

class PerimeterConstraint(HamiltonianConstraint):
    name: str = "PerimeterConstraint"
    lambda_perimeter: float = Field(1.0, description="Energy Multiplier for the Perimeter Hamiltonian Term")
    


    def delta(self, source, target, grid: "CPMGrid"):
        source_cell = grid.get_cell(source)
        target_cell = grid.get_cell(target)

        source_gain, target_gain = self.perimeter_change_if_copied(source=source, target=target, grid=grid)

        # Delta for the source cell (energy after gain - current energy)
        delta_h_s = self._h(source_cell, gain=source_gain) - self._h(source_cell, gain=0)
        # Delta for the target cell (energy after gain - current energy)
        delta_h_t = self._h(target_cell, gain=-target_gain) - self._h(target_cell, gain=0)

        return delta_h_s + delta_h_t

    def _h(self, cell: "CPMCell", gain:int):
        """
            Calculate the energy term given a cell and a gain in perimeter.
            That is, how far the perimeter will be to the target one if "gain" perimeter is added.
        """
        if cell.id == 0:
            # Background is always happy with its own perimeter.
            return 0
        
        perimeter_diff = cell.cell_type.preferred_perimeter - (cell.perimeter + gain)
        return self.lambda_perimeter * perimeter_diff * perimeter_diff


    def perimeter_change_if_copied(self, source, target, grid: "CPMGrid"):
        """
            Returns how the perimeter of the source and target cells would change if 
            source were copied to target.
        """
        # This code is similar to the one in copy_pixel but without the writes to neighbors.

        source_cell = grid.get_cell(source)
        target_cell = grid.get_cell(target)

        # Only consider the neighbors (all pixels) of the target cell (which is where change is happening)        
        neighbor_pixels =  grid.neighbors[target[0]][target[1]]

        if target_cell.id != 0:
            # New neighbors that would be created by giving up the target pixel 
            # (only pixels that belong to the target cell)
            nbs_to_add = sum([1 for nb in neighbor_pixels if grid.get_cell_id(nb) == target_cell.id])
            # Neighbors lost by the target cell by giving up the target pixel 
            # (the current neighbors of the target. aka, everything that is not the target cell)
            nbs_lost = len(target_cell._neighbors[tuple(target)])
            delta_perimeter_target = nbs_to_add - nbs_lost
        else:
            # Background doesn't have a "preferred perimeter", nor neighbours
            delta_perimeter_target = 0

        if source_cell.id != 0:
            # New neighbors that would be added to the source cell (everything that is not the source cell itself)
            new_nbs = sum([1 for nb in neighbor_pixels if grid.get_cell_id(nb) != source_cell.id])
            # Neighbors lost by extending the source_cell with the new pixel (evertything that is the source cell)
            nbs_removed = sum([1 for nb in neighbor_pixels if grid.get_cell_id(nb) == source_cell.id])
            delta_perimeter_source = new_nbs - nbs_removed
        else:
            delta_perimeter_source = 0

        return delta_perimeter_source, delta_perimeter_target