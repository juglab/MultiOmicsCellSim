import numpy as np
from pydantic import BaseModel, Field


class HamiltonianConstraint(BaseModel):
    """
        Represents an Hamiltonian term that imposes a constraint on the simulation
    """

    def delta(self, source, target, grid: "CPMGrid"):
        """
            Calculates the change in Hamiltonian energy (ΔH) for the target pixel 
            if the source pixel value were copied there.
        """
        # Initial adhesion energy at the target
        initial_energy = self.compute(source=source, target=target, grid=grid)

        # Simulate the copy operation (source -> target) to calculate potential new energy
        old_value = grid.get_pixel(target).copy()
        # FIXME: This is NOT thread safe!!!! But allows to simulate a whole step without copying the whole grid
        grid.copy_pixel(source, target)
        new_energy = self.compute(source=source, target=target, grid=grid)

        # Reset the target pixel to its original state
        grid.set_pixel(coords=target, value=old_value)
        
        return new_energy - initial_energy

class AdhesionConstraint(HamiltonianConstraint):
    """
        Pixels of the same cell will be constrained to stick together
    """



    def compute(self, source, target,  grid: "CPMGrid"):
        """
            Compute adhesion hamiltonian given a simulation state
        """

        # TODO: This does not work if celltypes are provided unsorted
        
        energy_matrix = np.stack([ct.adhesion_energy for ct in grid.cell_types])

        total_energy = 0
        
        #for x in range(grid.shape[0]):
        #    for y in range(grid.shape[0]):
        #mask = simulation_state._difference_mask(grid=grid, z_value=1)
        #mask_coords = list(zip(*np.where(mask)))

        # Only check source and target neighbors
        source_neighbors = grid.moore_neighborhood(*source)
        target_neighbors = grid.moore_neighborhood(*target)

        for x, y in source_neighbors + target_neighbors:
            neighbors_coords = grid.moore_neighborhood(x, y)
            source_celltype = grid.get_cell_type(source)
            for nx, ny in neighbors_coords:
                target_celltype = grid.get_cell_type([nx, ny])
                if source_celltype != target_celltype:
                    #print(f"{x=} {y=} {nx=} {ny=}")
                    # print(f"{x, y} vs {nx, ny}")
                    total_energy += energy_matrix[source_celltype, target_celltype]
        return total_energy

class VolumeConstraint(HamiltonianConstraint):
    lambda_volume: float = Field(1.0, description="Energy multiplier for the Volume Hamiltonian") 
    
    def compute(self, source, target,  grid: "CPMGrid"):
        source_cell_id = grid.get_cell_id(source)
        source_cell_type = grid.get_cell_type(source)
        target_cell_id = grid.get_cell_id(target)
        target_cell_type = grid.get_cell_type(target)

        total_energy = 0

        if source_cell_type != 0:
            source_pref_volume = grid.cell_types[source_cell_type].preferred_volume
            # FIXME: This is faster but approximate because it also counts the case a cell has splitted somehow
            # However, in case a cell split it should be assigned a new id so an appropriate check has to be made
            source_current_volume = np.sum(grid.mask_cell_id(source_cell_id))
            total_energy += self.lambda_volume * np.power(source_current_volume - source_pref_volume, 2)
        
        if target_cell_type != 0:
            target_pref_volume = grid.cell_types[target_cell_type].preferred_volume
            # FIXME: This is faster but approximate because it also counts the case a cell has splitted somehow
            # However, in case a cell split it should be assigned a new id so an appropriate check has to be made
            target_current_volume = np.sum(grid.mask_cell_id(target_cell_id))
            total_energy += self.lambda_volume * np.power(target_current_volume - target_pref_volume, 2)

        return total_energy