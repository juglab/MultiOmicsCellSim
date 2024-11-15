import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm

from tqdm import tqdm
import pandas as pd


class CPM():
    grid: "CPMGrid"

    def __init__(self, grid: "CPMGrid", debug: bool = False):
        self.grid = grid
        self.debug = debug
        self.stats = pd.DataFrame()

    def metropolis(self, step):

        """
        
        
        """

        # Avoid picking from inside the cells or in the background by picking only between cell frontiers
        
        if self.debug:
            for cell in self.grid._cells:
                cell.log(step=step)

        
        
        timestep = 0
        while timestep < 1.0:
            perimeters = [ int(c.perimeter) for c in self.grid._cells ]
            tot_perimeter = sum(perimeters)
            cell_pick_prob = [ p / tot_perimeter for p in perimeters]
            timestep += 1. / tot_perimeter

            new_copies = 0
            # Pick one random cell according to their perimeter length
            random_cell_id = np.random.choice(range(len(self.grid._cells)), p=cell_pick_prob)
            random_cell = self.grid._cells[random_cell_id]
            frontier_pxl, rand_neighbor = random_cell.get_random_neighboring_pair()
            if random.random() < .5:
                s_coord = list(frontier_pxl)
                t_coord = list(rand_neighbor)
            else:
                s_coord = list(rand_neighbor)
                t_coord = list(frontier_pxl)           

            s_cell_id = self.grid.get_cell_id(s_coord)
            t_cell_id = self.grid.get_cell_id(t_coord)

            if s_cell_id == t_cell_id:
                continue


            
            # Copy attempt
            ## Compute Hamiltonian for each constraint
            
            delta_h_i = [h.delta(s_coord, t_coord, self.grid) for h in self.grid.constraints]
            total_delta_energy = np.sum(delta_h_i)

            is_energy_decreasing = (total_delta_energy <= 0)
            boltzman_prob = np.exp(-total_delta_energy/self.grid.temperature)
            is_boltzman_passed = (np.random.uniform() <= boltzman_prob)

            # Attempt outcome
            if is_energy_decreasing or is_boltzman_passed:
                #print(f"{boltzman_prob=} {delta_h_i=} BOLTZMAN DEACTIVATED!!!!")
                self.grid.copy_pixel(source=s_coord, target=t_coord)
                new_copies += 1
                        
            if self.debug:
                new_stats = {
                        "step": [step],
                        "source_cell_id": [s_cell_id],
                        "target_cell_id": [t_cell_id],
                        "source_coords": [s_coord],
                        "target_coords": [t_coord],
                        "boltzman_prob": [boltzman_prob],
                        "boltzman_passed": [is_boltzman_passed],
                        "total_delta_h": [total_delta_energy],
                        "success": [is_energy_decreasing or is_boltzman_passed]
                    }
                constraint_names = [c.name for c in self.grid.constraints]
                
                for constr_name, constr_delta in zip(constraint_names, delta_h_i):
                    new_stats.update({constr_name: [constr_delta]})

                self.stats = pd.concat([self.stats, pd.DataFrame(new_stats)], ignore_index=True)        



    def step(self, max_steps=1):
        """ Perform n steps of Cellular Pott Model"""
        step_cache = [self.grid.grid.copy()]

        for s in tqdm(range(1, max_steps)):
            self.metropolis(step=s)
            step_cache.append(self.grid.grid.copy())

        return np.stack(step_cache)


    def render_animation(self, max_steps=10):
        """
        Displays a widget to play the full animation with controls.
        """
        # Generate and stack the animation frames
        step_cache = self.step(max_steps=max_steps)
                   
        # Set up the figure and subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for a in ax:
            a.axis('off')
        ax[0].set_title("Cell ID")
        ax[1].set_title("Cell Type")
        fig.tight_layout()
        
        # Apply the colormaps and display the first frame
        to_show_0 = ax[0].imshow(step_cache[0, ..., 0], animated=True, cmap="tab20b")
        to_show_1 = ax[1].imshow(step_cache[0, ..., 1], animated=True, cmap="tab20c")

        def animate(t):
            # Update images with the next frame
            to_show_0.set_data(step_cache[t, ..., 0])
            to_show_1.set_data(step_cache[t, ..., 1])
            return to_show_0, to_show_1
        
        anim = animation.FuncAnimation(fig, animate, frames=max_steps, blit=True)
        return anim
