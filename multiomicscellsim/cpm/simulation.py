import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm


class CPM():
    grid: "CPMGrid"

    def __init__(self, grid: "CPMGrid"):
        self.grid = grid

    def metropolis(self):
        # Avoid picking from inside the cells or in the background by picking only between cell frontiers
        
        mask = self.grid.get_frontier_mask(z_layer=0)

        mask_coord = list(zip(*np.where(mask)))

        stats = {'same_id': 0, 'failed': 0, 'passed': 0}

        for i in range(len(mask_coord)):
            rid = np.random.choice(range(len(mask_coord)))
            s_coord = mask_coord[rid]

            if not mask.any():
                print(f"Sim terminated, no more cell alive")
                return
            # s_*: source t_*: target
            #s_coord, (s_cell, s_type, s_sub) = self.sim.get_random_pixel()
            s_cell_id = self.grid.get_cell_id(s_coord)
            t_coord, (t_cell_id, t_type, t_sub) = self.grid.get_random_neighbour(source_coords=s_coord)

            if s_cell_id == t_cell_id:
                stats['same_id'] += 1
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
                stats['passed'] += 1
            else:
                stats['failed'] += 1


    def step(self, n=1):
        """ Perform n steps of Cellular Pott Model"""
        for n in range(n):
            #print(f"{n=}")
            self.metropolis()
        return self.grid.grid

    def render_animation(self, max_steps=10):
        """
        Displays a widget to play the full animation with controls.
        """
        # Generate and stack the animation frames
        
        step_cache = np.stack([self.grid.grid.copy()]+[self.step(1).copy() for _ in tqdm(range(max_steps))])
        
        # Set up the figure and subplots
        fig, ax = plt.subplots(1, 2)
        for a in ax:
            a.axis('off')
        ax[0].set_title("Cell ID")
        ax[1].set_title("Cell Type")
        fig.tight_layout()
        to_show_0 = ax[0].imshow(step_cache[0, ..., 0], animated=True)
        to_show_1 = ax[1].imshow(step_cache[0, ..., 1], animated=True)
        
        
        def animate(t):
            # Update images with the next frame
            to_show_0.set_data(step_cache[t, ..., 0])
            to_show_1.set_data(step_cache[t, ..., 1])
            return to_show_0, to_show_1
        
        anim = animation.FuncAnimation(fig, animate, frames=max_steps, blit=True)
        return anim 