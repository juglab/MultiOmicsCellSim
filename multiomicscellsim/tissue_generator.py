from .config import SimulatorConfig, TissueConfig, MicroscopySpaceConfig
from .entities import Tissue, Guideline, Cell, CellParams

import numpy as np
from scipy.stats.qmc import PoissonDisk

import matplotlib.pyplot as plt
from matplotlib import patches

import logging

from .utils.geometry import get_arcs_inside_rectangle, map_samples_to_arcs, circle_polar_to_cartesian
from .torch_cpm.simulation import TorchCPM

from typing import List
import random
from copy import deepcopy
from pathlib import Path
import yaml


logger = logging.getLogger(__name__)

class TissueGenerator():
    """
        Class for generating a tissue, taking as input a simulation configuration.
        Generates an initial state for the tissue and runs the cell growth algorithm to reach a final state.
    """
    tissue_config: TissueConfig
    microscopy_config: MicroscopySpaceConfig

    def __init__(self, simulator_config: SimulatorConfig):
        self.simulator_config = simulator_config
        self.tissue_config = simulator_config.tissue_config
        self.microscopy_config = simulator_config.microscopy_space_config
        self.cpm_config = simulator_config.cpm_config

    def _sample_guidelines(self, seed=None):
        """
            Sample a new set of guidelines.
            If tissue_config.allow_guideline_intersection is False, then uses a PoissonDisk distribution
            to keep circumferences apart.
            Circumferences are allowed to have their center outside of their images boundaries, 
            as long as "enough" of the circumference is shown, that is, the center must be not further
            from either axis than the guideline radius minus 3-sigma (which is where most of the cells will spawn).

            Args:
                - seed (int): Seed to use for the generation. If None, simulation is random.
                              This has to be passed as parameter wherever reproducibility is required
                              and not set via common methods since some internal functions does not
                              rely on math.random, numpy or torch for seed.
        """
        guidelines = []
        
        
        img_size = self.microscopy_config.coord_max - self.microscopy_config.coord_min
        # Radii in um
        min_radius = self.tissue_config.min_radius_perc * img_size
        max_radius = self.tissue_config.max_radius_perc * img_size
        
        # Buffer to allow guidelines centers to spawn outside of the microscopy space, 
        # but ensuring enough circumference is shown
        buffer = min_radius - 3 * self.tissue_config.guidelines_std
        logger.debug(f"{buffer=} {min_radius=} {max_radius=} {3 * self.tissue_config.guidelines_std=} ")
        extended_min = self.microscopy_config.coord_min - buffer
        extended_max = self.microscopy_config.coord_max + buffer
        logger.debug(extended_min, extended_max)


        if not self.tissue_config.allow_guideline_intersection:
            # The space sampled by PoissonDisk is defined in (0, 1)
            pd = PoissonDisk(d=2, radius=2*self.tissue_config.max_radius_perc, seed=self.simulator_config.simulator_seed if seed is None else seed)
            # Sampling centers in (0,1) and rescaling to microscopy coords
            centers = pd.random(self.tissue_config.n_curves) * (extended_max - extended_min) + extended_min
        else:
            centers = np.random.uniform(low=extended_min, 
                                        high=extended_max, 
                                        size=2)

        if len(centers) != self.tissue_config.n_curves:
            logger.warning(f"{self.tissue_config.n_curves} did not fit the drawing space \
                            and an image with {len(centers)} has been created. \
                            Please consider reducing max_radius_perc.")

        # Actual guideline radii
        guideline_radii = np.random.uniform(low=min_radius, 
                                            high=max_radius,
                                            size=len(centers)
                                            )
        
        for center, radius in zip(centers, guideline_radii):
            n_cells = np.round(np.random.normal(loc=self.tissue_config.cell_number_mean,
                                                scale=self.tissue_config.cell_number_std,
                                                )).astype(int)
            n_cells = max(1, n_cells)
            guideline = Guideline(
                            type="circle",
                            x_center=center[0],
                            y_center=center[1],
                            radius=radius,
                            radial_std=self.tissue_config.guidelines_std,
                            n_cells=n_cells
                        )
            guidelines.append(guideline)
        return guidelines
    
    def _sample_cell_centroid(self, guideline: Guideline):
        """
            Samples a cell centroid along a guideline
        """
        centroids = []
        # Avoid sampling over areas that are out of microscopy range
        # We only sample from the intervals of theta that correspond to arcs that lay within the image
        guideline_arcs = get_arcs_inside_rectangle(
                                                    xc=guideline.x_center,
                                                    yc=guideline.y_center,
                                                    rc=guideline.radius,
                                                    xr_min=self.microscopy_config.coord_min,
                                                    xr_max=self.microscopy_config.coord_max,
                                                    yr_min=self.microscopy_config.coord_min,
                                                    yr_max=self.microscopy_config.coord_max
        )
        # Sample from the circle, but only considering visible parts

        if guideline.tangent_distribution == "uniform":
            # Tangent samples are sampled in [0, 1]
            tangent_samples =np.random.uniform(size=[guideline.n_cells])
            # Mapping samples to the visible arcs
            sampled_thetas = map_samples_to_arcs(tangent_samples, arcs=guideline_arcs)
        else:
            raise NotImplementedError(f"{guideline.tangent_distribution} tangent distribution is not implemented")

        # Sample the radii corresponding to each theta
        if guideline.radial_distribution == "normal":
            sampled_radii = np.random.normal(loc=guideline.radius, scale=guideline.radial_std, size=guideline.n_cells)

        for theta, rad in zip(sampled_thetas, sampled_radii):
            coords = circle_polar_to_cartesian(theta=theta, xc=guideline.x_center, yc=guideline.y_center, r=rad)
            centroids.append(coords)

        return centroids
    
    def _cartesian_to_grid_coords(self, x, y):
        """
            Given some Cartesian coordinates (x, y), returns the row and column 
            coordinates of a grid of size self.cpm_grid_size.
        """
        # Define the bounds based on the microscopy configuration
        x_min = self.microscopy_config.coord_min
        x_max = self.microscopy_config.coord_max
        y_min = self.microscopy_config.coord_min
        y_max = self.microscopy_config.coord_max

        # Grid size
        grid_size = self.cpm_config.size

        # Normalize the coordinates to the range [0, grid_size - 1]
        col = int((x - x_min) / (x_max - x_min) * (grid_size - 1))
        row = int((y - y_min) / (y_max - y_min) * (grid_size - 1))

        # Clamp values to ensure they fall within grid bounds
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))

        return row, col

    def sample(self, tissue_id: int = 0, tissue_folder: Path = None, seed: int = None) -> List[Tissue]:
        """
            Sample a new tissue.

            Args: 
                tissue_id (int): An integer identifier for this tissue chain.
                tissue_folder (Path): A path where to save this tissue. If None, the tissue will be only returned.
        """

        # Define a CPM Simulation
        cpm = TorchCPM(self.cpm_config)

        # Generate the guidelines (in microscopy space) for spawning cells
        guidelines = self._sample_guidelines(seed=seed)
        cells = []
        for g, gl in enumerate(guidelines):
            centroids = self._sample_cell_centroid(gl)

            cpm_cell_centroids = [self._cartesian_to_grid_coords(c[0], c[1]) for c in centroids]

            # Spawn cells in the CPM grid
            for centroid, cpm_cell_coord in zip(centroids, cpm_cell_centroids):
                cell_type = random.choices(population=self.cpm_config.cell_types,
                                           weights=self.tissue_config.cell_type_probabilities[g])[0]
                cell_id = cpm.draw_cell(x=cpm_cell_coord[0], 
                                          y=cpm_cell_coord[1], 
                                          cell_type=cell_type.id,
                                          size=self.tissue_config.initial_cell_size,
                                          )
                # cell_id is 0 if write failed
                if cell_id > 0:
                    cells.append(Cell(cell_id=cell_id,
                                      cell_type=cell_type,
                                      start_coordinates_cpm = cpm_cell_coord,
                                      start_coordinates=centroid,
                                      )
                                 )
                    gl.spawned_cell_ids.append(cell_id)

        tissues = list()

        # Grow Cells and save a tissue at predetermined steps
        for cell_grid, subcell_grid, cpm_steps, rd_steps in cpm.yield_step(yield_every=self.simulator_config.save_tissue_every):

            cells = self._extract_cells_params_from_grid(cells, cell_grid, subcell_grid)
            tissue = Tissue(
                        id=tissue_id,
                        tissue_seed=seed,
                        cpm_step=cpm_steps,
                        rd_step=rd_steps,
                        guidelines=guidelines,
                        cells=cells,
                        cell_grid=cell_grid,
                        subcell_grid=subcell_grid
                    )
            # Update cell representation from image (parameter update)
            tissues.append(tissue)

            if tissue_folder is not None:
                tissue.save(tissue_folder=tissue_folder)
                logger.info(f"Tissue exported to {tissue_folder}")
        return tissues
    
    def _extract_cells_params_from_grid(self, cells: List[Cell], cell_grid: np.ndarray, subcell_grid: np.ndarray):
        """
            Given a list of Cell, updates its parameters from a given (sub)cell_grid.
        """

        for cell in cells:

            cell_mask = (cell_grid[0]==cell.cell_id)
            A_subcellular = subcell_grid[0][cell_mask].numpy()
            B_subcellular = subcell_grid[1][cell_mask].numpy()

            # TODO: For now we are taking the params directly from the cell type itself.
            # If we implement some cell variability we should take it from wherever we store (maybe a pre-existing params value?)
            cell.params = CellParams(
                f=cell.cell_type.subcellular_pattern.f,
                k=cell.cell_type.subcellular_pattern.k,
                d_a=cell.cell_type.subcellular_pattern.d_a,
                d_b=cell.cell_type.subcellular_pattern.d_b,
                a_avg=A_subcellular.mean(),
                a_std=A_subcellular.std(),
                b_avg=B_subcellular.mean(),
                b_std=B_subcellular.std()
            )
        return deepcopy(cells)

    def plot_debug(self, tissue: Tissue, size: int = 8):
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # Setup axes in microscopy space
        #ax.invert_yaxis()
        ax.set_xlim(0, self.microscopy_config.coord_max)
        ax.set_ylim(0, self.microscopy_config.coord_max)
        
        # Plotting guidelines
        # Plotting centers
        x_centers = [gl.x_center for gl in tissue.guidelines]
        y_centers = [gl.y_center for gl in tissue.guidelines]
        radii = [gl.radius for gl in tissue.guidelines]
        plt.scatter(x=x_centers, y=y_centers)
        # Plotting guidelines
        for x, y, r in zip(x_centers, y_centers, radii):
            circle = patches.Circle(xy=[x, y], radius=r, fill=False)
            ax.add_patch(circle)
            max_circle = patches.Circle(xy=[x, y], radius=self.tissue_config.max_radius_perc*self.microscopy_config.coord_max, fill=False, linestyle="dashed", edgecolor="red")
            ax.add_patch(max_circle)
            min_circle = patches.Circle(xy=[x, y], radius=self.tissue_config.min_radius_perc*self.microscopy_config.coord_max, fill=False, linestyle="dashed", edgecolor="green")
            ax.add_patch(min_circle)

        # Plotting cells
        for cell in tissue.cells:
            x, y = cell.start_coordinates
            # Plot cell position as an "X"
            ax.plot(x, y, 'x', color='blue')

        plt.gca().invert_yaxis()
        plt.show()

