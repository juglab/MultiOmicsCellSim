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

def vectorized_moore_neighborhood(x:torch.Tensor, neighborhood_type=8, background_value=-1, cell_id_channel=0):
    """
        This is an vectorized implementation of the Moore neighborhood.
        The neighbor value in each direction is stored in a separate channel c.
        Channels are ordered clockwise starting from the top left corner if 
        neighborhood_type is 8, or starting from the top if neighborhood_type is 4.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w] or [l, h, w].
            neighborhood_type (int): Type of neighborhood to use. Can be 4 or 8.
            background_value (int): Value to use for the background.
            cell_id_channel (int): Channel to use for the cell id if x is a tensor of shape [l, h, w].
        
        Returns:
            torch.Tensor: Tensor of shape [c, h, w] containing the neighbors value of each pixel.

    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    if neighborhood_type == 8:
        shifts = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
    else:
        shifts = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    # Pad -> Roll in every direction -> Unpad -> Stack
    return torch.stack([torch.roll(F.pad(x, (1, 1, 1, 1), value=background_value), shifts=shift, dims=(0, 1))[1:-1, 1:-1] for shift in shifts], dim=0)
    
def get_frontiers(x:torch.Tensor, cell_id_channel=0):
    """
        Returns a mask of the inner perimeters of the objects in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w] or [l, h, w].
            cell_id_channel (int): Channel to use for the cell id if x is a tensor of shape [l, h, w].
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    n = vectorized_moore_neighborhood(x)
    return torch.any(n != x.unsqueeze(0), dim=0)

def get_differet_neigborhood_mask(x:torch.Tensor, cell_id_channel=0) -> torch.Tensor:
    """
        Given a tensor x of shape [h, w], or [l, h, w], returns a tensor of shape [8, h, w] where each channel c
        contains the mask of the neighbors of each pixel in x that are different from the pixel itself.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w] or [l, h, w].
            cell_id_channel (int): Channel to use for the cell id if x is a tensor of shape [l, h, w].
        
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    n = vectorized_moore_neighborhood(x)
    return n != x.unsqueeze(0)

def get_different_neighbors(x:torch.Tensor, cell_id_channel=0):
    """
        Returns a tensor of shape [8, h, w] where each channel c contains the neighbors of each pixel in x that are different from the pixel itself.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    n = vectorized_moore_neighborhood(x)
    return (n != x.unsqueeze(0)).int()*n

def get_different_neigbor_sum(x:torch.Tensor, cell_id_channel=0):
    """
        Returns a map that for each pixel stores the number of different neighbors.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w].

        Returns:
            torch.Tensor: Tensor of shape [h, w] containing the number of different neighbors for each pixel
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    n = vectorized_moore_neighborhood(x)
    return torch.sum(n != x.unsqueeze(0), dim=0)

def sum_over_objects(x:torch.Tensor, neighbors: torch.Tensor, cell_id_channel=0):
    """
        Returns a map that for each pixel store the sum of the neighbors values (along the first dimension) of the object it belongs to.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    sums = torch.zeros_like(x, torch.float)
    for obj in torch.unique(x):
        if obj == 0:
            continue
        sums[x == obj] = neighbors[x == obj].sum(dim=0)
    return sums

def get_volume_map(x:torch.Tensor, cell_id_channel=0):
    """
        Return a map that for each pixel store the volume of the object it belongs to.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]
    volumes = torch.zeros_like(x)
    for obj in torch.unique(x):
        if obj <= 0:
            continue
        volumes[x == obj] = (x == obj).sum()
    return volumes

def get_volume(x:torch.Tensor, cell_id_channel=0):
    """
        Returns the total volume of each object in the input tensor.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]
    volumes = torch.zeros(size=[x.unique().max()+1])

    for obj in torch.unique(x):
        if obj == 0:
            continue
        volumes[obj] = (x == obj).sum()
    return volumes

def get_perimeter_map(x:torch.Tensor, current_neighbors: torch.Tensor=None, cell_id_channel=0):
    """
        Calculate the perimeter for each different object in the given tensor.
        If current_neighbors is provided, computation is skipped.

        Args:
            x: [H, W], v: -1 for background, >0 for object masks
            current_neigbors (optional): [8, H, W], v: -1 for background, >0 for object masks
        Returns:
            A tensor [H, W] in which each object in x is assigned the value of its perimeter (for every pixel).
    """
    if x.ndim == 3:
        x = x[cell_id_channel]
    if current_neighbors is None:
        current_neighbors = vectorized_moore_neighborhood(x)
    perimeters = torch.zeros_like(x)
    pixelwise_perimeters = (current_neighbors != x.unsqueeze(0)).sum(dim=0)
    for obj in torch.unique(x):
        if obj <= 0:
            continue
        perimeters[x == obj] = pixelwise_perimeters[x == obj].sum()
    return perimeters

def get_adhesion_map(x:torch.Tensor, current_neighbors: torch.Tensor, adhesion_matrix: torch.Tensor, cell_id_channel=0):
    """
        Gets the adhesion map for each object in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [h, w].
            adhesion_matrix (torch.Tensor): Tensor of shape [n, n] containing the adhesion values between the different cell types.
        Returns:
            torch.Tensor: Tensor of shape [h, w] containing the adhesion values for each object.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]
    adhesions = torch.zeros_like(current_neighbors, dtype=torch.float)
    for s_val in torch.unique(x):
        for t_val in torch.unique(current_neighbors):
            if s_val == 0 or t_val == 0:
                continue
            s_idx = 0 if s_val < 0 else s_val
            t_idx = 0 if t_val < 0 else t_val
            adhesions[(x.unsqueeze() == s_val)&(current_neighbors == t_val)] = adhesion_matrix[s_idx, t_idx]
    return adhesions

def map_value_to_objects(x:torch.Tensor, values:torch.Tensor, cell_id_channel=0):
    """
        Given a tensor x (either in shape [h, w] or [c, h, w]) containing objects ids and a tensor values of len(torch.unique(x)),
        returns a tensor of the same shape as x where each object is assigned the value
        corresponding to its label in values.
    """
    if x.ndim == 3:
        x = x[cell_id_channel]

    result = torch.zeros_like(x, dtype=torch.float)
    for obj in torch.unique(x):
        if obj == 0 or obj == -1:
            continue
        result[x == obj] = values[obj]
    return result
  
def copy_source_to_neighbors(x: torch.Tensor, source_pixels: torch.Tensor, selected_neighbors: torch.Tensor, padding_value=-1):
    """
        Given an image img of shape [h, w] or [l, h, w], a tensor source_pixels of shape [h, w] containing the source pixels values to copy,
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
    dims = (0, 1) if x.ndim == 2 else (1, 2)

    for d in range(8):
        # Roll the original image to match the neigbor direction
        x = torch.roll(F.pad(x, (1, 1, 1, 1), value=padding_value), shifts=shifts[d], dims=dims)[..., 1:-1, 1:-1] 
        # Copy the source pixels to the selected neighbors
        x = torch.where(selected_neighbors[d] != 0, source_pixels, x)
        # Roll back the image to the original position
        x = torch.roll(F.pad(x, (1, 1, 1, 1), value=padding_value), shifts=[-shifts[d][0], -shifts[d][1]], dims=dims)[..., 1:-1, 1:-1]
    return x

def copy_subgrid_to_neighbors(s: torch.Tensor, selected_neighbors: torch.Tensor, padding_value=0):
    """
        Given a subgrid of floating points and shape [2, h, w], and a binary mask of tensor selected_neighbors of shape [8, h, w] containing the neighbors to copy the source pixels to,
        returns a new subgrid with the source pixels copied to the selected neighbors where the selected neighbors are True.

        Args:
            s (torch.Tensor): Input tensor of shape [2, h, w].
            selected_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors to copy the source pixels to.
                                               The dimension should be reported clockwise starting from the top left corner.

        Returns:
            torch.Tensor: Tensor of shape [2, h, w] containing the new subgrid with the source pixels copied to the selected neighbors
    """

    shifts = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
    dims = (1, 2)
    original = s.clone()
    for d in range(8):
        # Roll the original image to match the neigbor direction
        s = torch.roll(F.pad(s, (1, 1, 1, 1), value=padding_value), shifts=shifts[d], dims=dims)[..., 1:-1, 1:-1] 
        # Copy the source pixels to the selected neighbors
        s = torch.where(selected_neighbors[d].bool(), s, original)
        # Roll back the image to the original position
        s = torch.roll(F.pad(s, (1, 1, 1, 1), value=padding_value), shifts=[-shifts[d][0], -shifts[d][1]], dims=dims)[..., 1:-1, 1:-1]
    return s


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

def smallest_square_crop(mask, hexagon_size=None):
    """
    Crops the smallest square region containing all non-zero elements in the mask.

    Args:
        mask (torch.Tensor): 2D binary mask with non-zero values defining the region of interest.
        hexagon_size (int): The size (radius) of the hexagon, if cells are hexagonal.

    Returns:
        cropped_mask (torch.Tensor): Cropped square region of the mask.
        bounds (tuple): Bounding box (start_row, end_row, start_col, end_col).
    """
    # Find non-zero indices in the mask
    non_zero_indices = mask.nonzero(as_tuple=True)
    if len(non_zero_indices[0]) == 0:
        raise ValueError("Mask has no non-zero elements to crop.")

    # Compute bounding box dimensions
    min_row, max_row = non_zero_indices[0].min().item(), non_zero_indices[0].max().item()
    min_col, max_col = non_zero_indices[1].min().item(), non_zero_indices[1].max().item()

    # If hexagon_size is specified, calculate square side dynamically
    if hexagon_size:
        square_side = int(2 * hexagon_size)  # Side of the square bounding box
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2

        # Adjust bounds to ensure the square fits the hexagon
        min_row = max(0, center_row - square_side // 2)
        max_row = min(mask.shape[0], center_row + square_side // 2)
        min_col = max(0, center_col - square_side // 2)
        max_col = min(mask.shape[1], center_col + square_side // 2)

    # Crop the mask
    cropped_mask = mask[min_row:max_row, min_col:max_col]

    return cropped_mask, (min_row, max_row, min_col, max_col)
