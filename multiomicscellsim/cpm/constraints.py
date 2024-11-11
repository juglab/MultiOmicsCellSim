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

from skimage.measure import perimeter

class PerimeterConstraint(HamiltonianConstraint):
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
        """
        if cell.id == 0:
            return 0
        
        perimeter_diff = cell.cell_type.preferred_perimeter - (cell.perimeter + gain)
        return self.lambda_perimeter * perimeter_diff * perimeter_diff


    def perimeter_change_if_copied(self, source, target, grid: "CPMGrid"):
        """
            Returns how the perimeter of the source and target cells would change if 
            source were copied to target.
        """
        # This code is similar to the one in copy_pixel but without the writes to neighbors.

        delta_perimeter_source = 0
        delta_perimeter_target = 0

        source_cell = grid.get_cell(source)
        target_cell = grid.get_cell(target)

        # update neighbors and perimeters
        new_nbs = [] # new row in neighbors table for the source cell
        for nb in grid.neighbors[target[0]][target[1]]:
            nb_cell_id = grid.get_cell_id(nb)
            if nb_cell_id == source_cell.id:
                if source_cell.id != 0:
                    delta_perimeter_source -= 1
            else:
                new_nbs.append(nb)
                if nb_cell_id != 0 and target_cell.id !=0:
                    delta_perimeter_target += 1

        if source_cell.id != 0:
            delta_perimeter_source += len(new_nbs)
        
        if target_cell.id != 0:
            # Remove the neighbors of the target pixels formerly belonging to the removed pixel
            delta_perimeter_target -= len(target_cell._neighbors[tuple(target)])
        return delta_perimeter_source, delta_perimeter_target

    # def compute(self, source, target, grid:"CPMGrid"):
    #     source_cell_id = grid.get_cell_id(source)
    #     source_cell_type = grid.get_cell_type(source)
    #     target_cell_id = grid.get_cell_id(target)
    #     target_cell_type = grid.get_cell_type(target)

    #     total_energy = 0

    #     if source_cell_type != 0:
    #         s_pref_perim = grid.cell_types[source_cell_type].preferred_perimeter
    #         m_s = grid.mask_cell_id(source_cell_id)
    #         p_s = perimeter(image=m_s, neighborhood=8 if grid.neighborhood == 'moore' else 4)
    #         total_energy += self.lambda_perimeter * np.power(p_s - s_pref_perim, 2)
        
    #     if target_cell_type != 0:
    #         t_pref_perim = grid.cell_types[target_cell_type].preferred_perimeter
    #         m_t = grid.mask_cell_id(target_cell_id)
    #         p_t = perimeter(image=m_t, neighborhood=8 if grid.neighborhood == 'moore' else 4)
    #         total_energy += self.lambda_perimeter * np.power(p_t - t_pref_perim, 2)

    #     return total_energy