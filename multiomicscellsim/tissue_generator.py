from .config import SimulatorConfig, TissueConfig, MicroscopySpaceConfig
from .entities import Tissue, Guideline

import numpy as np
from scipy.stats.qmc import PoissonDisk

import matplotlib.pyplot as plt
from matplotlib import patches

import logging

from .utils.geometry import get_arcs_inside_rectangle

logger = logging.getLogger(__name__)

class TissueGenerator():
    tissue_config: TissueConfig
    microscopy_config: MicroscopySpaceConfig

    def __init__(self, simulator_config: SimulatorConfig):
        self.tissue_config = simulator_config.tissue_config
        self.microscopy_config = simulator_config.microscopy_space_config

    def _sample_guidelines(self):
        """
            Sample a new set of guidelines.
            If tissue_config.allow_guideline_intersection is False, then uses a PoissonDisk distribution
            to keep circumferences apart.
            Circumferences are allowed to have their center outside of their images boundaries, 
            as long as "enough" of the circumference is shown, that is, the center must be not further
            from either axis than the guideline radius minus 3-sigma (which is where most of the cells will spawn).
                    
        """
        guidelines = []
        
        
        img_size = self.microscopy_config.coord_max - self.microscopy_config.coord_min
        # Radii in um
        min_radius = self.tissue_config.min_radius_perc * img_size
        max_radius = self.tissue_config.max_radius_perc * img_size
        
        

        # Buffer to allow guidelines centers to spawn outside of the microscopy space, 
        # but ensuring enough circumference is shown
        buffer = min_radius - 3 * self.tissue_config.guidelines_std
        print(f"{buffer=} {min_radius=} {max_radius=} {3 * self.tissue_config.guidelines_std=} ")
        extended_min = self.microscopy_config.coord_min - buffer
        extended_max = self.microscopy_config.coord_max + buffer
        print(extended_min, extended_max)


        if not self.tissue_config.allow_guideline_intersection:
            # The space sampled by PoissonDisk is defined in (0, 1)
            pd = PoissonDisk(d=2, radius=2*self.tissue_config.max_radius_perc)
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
            guideline = Guideline(
                            type="circle",
                            x_center=center[0],
                            y_center=center[1],
                            radius=radius,
                            variance=self.tissue_config.guidelines_std,
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
        guideline_arcs = get_arcs_inside_rectangle(
                                                    xc=guideline.x_center,
                                                    yc=guideline.y_center,
                                                    rc=guideline.radius,
                                                    xr_min=self.microscopy_config.coord_min,
                                                    xr_max=self.microscopy_config.coord_max,
                                                    yr_min=self.microscopy_config.coord_min,
                                                    yr_max=self.microscopy_config.coord_max
        )

        return centroids

    def sample(self):
        """
            Sample a new tissue.
        """
        # Generate guidelines
        guidelines = self._sample_guidelines()
        for gl in guidelines:
            centroids = self._sample_cell_centroid(gl)


        return Tissue(
            guidelines=guidelines
        )


    def render(self):
        """
            Rasterize an image representation of the tissue
        """
        pass


    def plot_debug(self, tissue: Tissue, size: int = 8):
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # Setup axes in microscopy space
        ax.invert_yaxis()
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

        plt.show()

