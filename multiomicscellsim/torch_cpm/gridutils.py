import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_tensors(tensors: list, titles: list = None, cmap='viridis'):
    """
        Plots a list of tensors.
        
        Args:
            tensors (list): List of tensors to plot.
            titles (list): List of titles for each tensor.
            cmap (str): Matplotlib colormap to use.
    """
    fig, axs = plt.subplots(1, len(tensors), figsize=(15*len(tensors), 5*len(tensors)))
    if len(tensors) == 1:
        axs = [axs]
    for i, tensor in enumerate(tensors):
        #axs[i].pcolormesh(tensor,  edgecolors='k', linewidth=0.1)
        axs[i].imshow(tensor, cmap=cmap)
        if titles is not None:
            axs[i].set_title(titles[i])
        #axs[i].axis('off')
        # Add independent colorbar for each tensor
        fig.colorbar(axs[i].imshow(tensor, cmap=cmap), ax=axs[i])

    fig.tight_layout()
    plt.show()

def vectorized_moore_neighborhood(x:torch.Tensor, neighborhood_type=8, background_value=-1):
    """
        This is an vectorized implementation of the Moore neighborhood.
        The neighbor value in each direction is stored in a separate channel c.
        Channels are ordered clockwise starting from the top left corner if 
        neighborhood_type is 8, or starting from the top if neighborhood_type is 4.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w].
            neighborhood_type (int): Type of neighborhood to use. Can be 4 or 8.
        
        Returns:
            torch.Tensor: Tensor of shape [c, h, w] containing the neighbors value of each pixel.

    """
    if neighborhood_type == 8:
        shifts = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
    else:
        shifts = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    # Pad -> Roll in every direction -> Unpad -> Stack
    return torch.stack([torch.roll(F.pad(x, (1, 1, 1, 1), value=background_value), shifts=shift, dims=(0, 1))[1:-1, 1:-1] for shift in shifts], dim=0)
    
def get_frontiers(x:torch.Tensor):
    """
        Returns a mask of the inner perimeters of the objects in the input tensor.
    """
    n = vectorized_moore_neighborhood(x)
    return torch.any(n != x.unsqueeze(0), dim=0)

def get_differet_neigborhood_mask(x:torch.Tensor) -> torch.Tensor:
    """
        Given a tensor x of shape [h, w], returns a tensor of shape [8, h, w] where each channel c
        contains the mask of the neighbors of each pixel in x that are different from the pixel itself.
    """
    n = vectorized_moore_neighborhood(x)
    return n != x.unsqueeze(0)

def get_different_neighbors(x:torch.Tensor):
    """
        Returns a tensor of shape [8, h, w] where each channel c contains the neighbors of each pixel in x that are different from the pixel itself.
    """
    n = vectorized_moore_neighborhood(x)
    return (n != x.unsqueeze(0)).int()*n

def get_different_neigbor_sum(x:torch.Tensor):
    """
        Returns a map that for each pixel stores the number of different neighbors.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w].

        Returns:
            torch.Tensor: Tensor of shape [h, w] containing the number of different neighbors for each pixel
    """
    n = vectorized_moore_neighborhood(x)
    return torch.sum(n != x.unsqueeze(0), dim=0)

def sum_over_objects(x:torch.Tensor, neighbors: torch.Tensor):
    """
        Returns a map that for each pixel store the sum of the neighbors values (along the first dimension) of the object it belongs to.
    """
    sums = torch.zeros_like(x, torch.float)
    for obj in torch.unique(x):
        if obj == 0:
            continue
        sums[x == obj] = neighbors[x == obj].sum(dim=0)
    return sums

def get_volume_map(x:torch.Tensor):
    """
        Return a map that for each pixel store the volume of the object it belongs to.
    """
    volumes = torch.zeros_like(x)
    for obj in torch.unique(x):
        if obj <= 0:
            continue
        volumes[x == obj] = (x == obj).sum()
    return volumes

def get_volume(x:torch.Tensor):
    """
        Returns the total volume of each object in the input tensor.
    """
    volumes = torch.zeros(size=[x.unique().max()+1])

    for obj in torch.unique(x):
        if obj == 0:
            continue
        volumes[obj] = (x == obj).sum()
    return volumes

def get_perimeter_map(x:torch.Tensor, current_neighbors: torch.Tensor=None):
    """
        Calculate the perimeter for each different object in the given tensor.
        If current_neighbors is provided, computation is skipped.

        Args:
            x: [H, W], v: -1 for background, >0 for object masks
            current_neigbors (optional): [8, H, W], v: -1 for background, >0 for object masks
        Returns:
            A tensor [H, W] in which each object in x is assigned the value of its perimeter (for every pixel).
    """
    if current_neighbors is None:
        current_neighbors = vectorized_moore_neighborhood(x)
    perimeters = torch.zeros_like(x)
    pixelwise_perimeters = (current_neighbors != x.unsqueeze(0)).sum(dim=0)
    for obj in torch.unique(x):
        if obj <= 0:
            continue
        perimeters[x == obj] = pixelwise_perimeters[x == obj].sum()
    return perimeters

def get_adhesion_map(x:torch.Tensor, current_neighbors: torch.Tensor, adhesion_matrix: torch.Tensor):
    """
        Gets the adhesion map for each object in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w].
            adhesion_matrix (torch.Tensor): Tensor of shape [n, n] containing the adhesion values between the different cell types.
        Returns:
            torch.Tensor: Tensor of shape [h, w] containing the adhesion values for each object.
    """
    adhesions = torch.zeros_like(current_neighbors, dtype=torch.float)
    for s_val in torch.unique(x):
        for t_val in torch.unique(current_neighbors):
            if s_val == 0 or t_val == 0:
                continue
            s_idx = 0 if s_val < 0 else s_val
            t_idx = 0 if t_val < 0 else t_val
            adhesions[(x.unsqueeze() == s_val)&(current_neighbors == t_val)] = adhesion_matrix[s_idx, t_idx]
    return adhesions

def map_value_to_objects(x:torch.Tensor, values:torch.Tensor):
    """
        Given a tensor x (either in shape [h, w] or [c, h, w]) containing objects ids and a tensor values of len(torch.unique(x)),
        returns a tensor of the same shape as x where each object is assigned the value
        corresponding to its label in values.
    """
    result = torch.zeros_like(x, dtype=torch.float)
    for obj in torch.unique(x):
        if obj == 0 or obj == -1:
            continue
        result[x == obj] = values[obj]
    return result


def calc_energy_delta_volume(source_pixels: torch.Tensor, target_pixels: torch.Tensor, preferred_volumes: torch.Tensor):
    """
        Calculates the energy difference of the system if the source pixels are changed to the target pixels.
    """

    def h(volumes, preferred_volumes, gains, lambda_volume=1.0):
        """
            Energy function that penalizes the difference between the current volume and the preferred volume
        """
        v_diff = preferred_volumes - (volumes + gains)
        return lambda_volume*(v_diff**2)

    current_volume_source = get_volume_map(source_pixels)
    current_volume_target = get_volume_map(target_pixels)
    preferred_volumes_source = map_value_to_objects(source_pixels, preferred_volumes)
    preferred_volumes_target = map_value_to_objects(target_pixels, preferred_volumes)

    # Calculate the volume gain based on source and target pixels
    # The cell of source pixels are expanding, so they get one more volume unit
    delta_sources = h(current_volume_source, preferred_volumes_source, 1) - h(current_volume_source, preferred_volumes_source, 0)
    # The cell of target pixels are shrinking, so they lose one volume unit
    delta_targets = h(current_volume_target, preferred_volumes_target, -1) - h(current_volume_target, preferred_volumes_target, 0)

    return delta_sources + delta_targets




def calc_energy_delta_local_perimeter(current_state: torch.Tensor, current_diff_nbs: torch.Tensor, predicted_state: torch.Tensor, predicted_diff_nbs: torch.Tensor, preferred_perimeters: torch.Tensor):
    """
        Imposes a local perimeter constraint on each border pixel. It can be used to control the roughness of the object borders.

    """

    def h(current_perimeters: torch.Tensor, preferred_perimeters: torch.Tensor, lambda_perimeter=1.0):
        """
            Function that calculate the energy of the system based on the perimeter of each object.
        """
        return lambda_perimeter * (preferred_perimeters - current_perimeters)**2
    
    # TODO: Find a way to compute a local energy, for now it is computed globally so each pixel is given the same energy in the output

    # Perimeter contribution of each pixel
    curr_local_perimeter = current_diff_nbs.sum(dim=0)
    # Perimeter contribution of each pixel
    pred_local_perimeter = predicted_diff_nbs.sum(dim=0)
    
    preferred_source_perimeters = map_value_to_objects(current_state, preferred_perimeters)
    preferred_target_perimeters = map_value_to_objects(predicted_state, preferred_perimeters)
    return h(curr_local_perimeter, preferred_source_perimeters) - h(pred_local_perimeter, preferred_target_perimeters)


  
def copy_source_to_neighbors(x: torch.Tensor, source_pixels: torch.Tensor, selected_neighbors: torch.Tensor):
    """
        Given an image img of shape [h, w], a tensor source_pixels of shape [h, w] containing the source pixels values to copy,
        and a tensor selected_neighbors of shape [8, h, w] containing the neighbors to copy the source pixels to,
        returns a new image with the source pixels copied to the selected neighbors where the selected neighbors are differnt from zero.

        Args:
            img (torch.Tensor): Input tensor of shape [h, w].
            source_pixels (torch.Tensor): Tensor of shape [h, w] containing the source pixels values.
            selected_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors to copy the source pixels to.
                                               The dimension should be reported clockwise starting from the top left corner.

        Returns:
            next_state - torch.Tensor: Tensor of shape [h, w] containing the new image with the source pixels copied to the selected neighbors
            diff_neighbors - torch.Tensor: Tensor of shape [8, h, w] containing the neighbors values of each pixel
    """

    shifts = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
    for d in range(8):
        # Roll the original image to match the neigbor direction
        x = torch.roll(F.pad(x, (1, 1, 1, 1), value=-1), shifts=shifts[d], dims=(0, 1))[1:-1, 1:-1]
        # Copy the source pixels to the selected neighbors
        x = torch.where(selected_neighbors[d] != 0, source_pixels, x)
        # Roll back the image to the original position
        x = torch.roll(F.pad(x, (1, 1, 1, 1), value=-1), shifts=[-shifts[d][0], -shifts[d][1]], dims=(0, 1))[1:-1, 1:-1]
    return x, get_different_neighbors(x)


def choose_random_neighbor(x: torch.Tensor):
    """
    Given a tensor of shape (C, H, W), where the C dimension represents suitable 
    neighbors to be chosen for a given H,W pixel (i.e., value > 0),
    returns a tensor of the same shape of the input, 
    having only one value at 1.0 for each H,W location, 
    choosen randomly from the allowed pixels.
    Please notice that to ensure equal probability, a binary input is expected.
    Otherwise, the choice in the C dimension will be weighted by the input value.

    USAGE: If the input has a mask to be applied (either in H, W, or C dimension, apply it before 
    calling the function. I.e., if some pixels are to be excluded, set the corresponding C dimension to 0.

    """

    C, H, W = x.shape
    # Probs is a matrix with pixels as rows and neighbor probabilities as cols
    probs = x.permute(1, 2, 0).view([H*W,C]).float()
    # Find the pixels where all channels are False  (they would break multinomial)
    allzero_mask = torch.all(probs == 0, dim=(1))
    # Set to a random value to avoid breaking multinomial. Will be masked out of the final output
    probs[allzero_mask] = 0.5
    # Get a map (H,W) with chosen neighbor index as value
    rand_nbr_idxs = torch.multinomial(probs, 1).view([H, W])
    # Result is a clean map with True only where neighbor is chosen
    chosen_nbr = torch.zeros_like(x)
    chosen_nbr[rand_nbr_idxs, torch.arange(H).unsqueeze(1), torch.arange(W).unsqueeze(0)] = 1.0
    # Remove choices generated from multinomial workaround
    chosen_nbr[:, allzero_mask.view([H, W])] = 0.0
    return chosen_nbr.bool()

def smallest_square_crop(mask):
    # Find non-zero mask indices
    non_zero_indices = mask.nonzero(as_tuple=True)  # tuple of row and col indices
    min_row, max_row = non_zero_indices[0].min(), non_zero_indices[0].max()
    min_col, max_col = non_zero_indices[1].min(), non_zero_indices[1].max()
    
    # Calculate bounding box width and height
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    
    # Determine the size of the square
    square_size = max(width, height)
    
    # Calculate the center of the bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    
    # Adjust the start and end indices to create a square crop
    half_size = square_size // 2
    start_row = max(0, center_row - half_size)
    end_row = start_row + square_size
    start_col = max(0, center_col - half_size)
    end_col = start_col + square_size
    
    # Adjust for boundary conditions if the square exceeds the original image dimensions
    if end_row > mask.shape[0]:
        start_row = mask.shape[0] - square_size
        end_row = mask.shape[0]
    if end_col > mask.shape[1]:
        start_col = mask.shape[1] - square_size
        end_col = mask.shape[1]
    
    # Ensure valid start indices
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    
    # Crop the mask
    cropped_mask = mask[start_row:end_row, start_col:end_col]
    
    # Return the cropped mask and the bounding box indices
    return cropped_mask, (start_row, end_row, start_col, end_col)