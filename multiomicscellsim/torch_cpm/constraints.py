import torch
from .config import TorchCPMConfig
from .gridutils import get_volume_map, map_value_to_objects

class TorchCPMConstrint:
    
    def __init__(self, cpm_config: TorchCPMConfig):
        self.config = cpm_config

class TorchCPMAdhesionConstraint(TorchCPMConstrint):
    """
        Energy coefficient that represent the tendency of different cell types to stick together.
    """
    def __init__(self, cpm_config: TorchCPMConfig):
        super().__init__(cpm_config=cpm_config)

    def h(self, source_pixels, current_neighbors, adhesion_matrix):
        """
            Local energy is given by the sum of the adhesion matrix values between the neighbors (that are not the current one).
            This function simulate the adhesion of every input pixel as if they were changed to a given id.
        """
        source_pixels = source_pixels.int()
        current_neighbors = current_neighbors.int()
        # Store the adhesion coefficients for each src_pixel -> target_pixel pair
        adhesion_coeffs = torch.zeros_like(current_neighbors)

        # current_neighbors contains the cell_id, not cell_type
        # Boundaries between different CELLS
        diff_nbs_cell_id = current_neighbors*(current_neighbors != source_pixels[0].unsqueeze(0))
        # Map each cell_id to its cell_type
        diff_nbs_cell_type = torch.zeros_like(diff_nbs_cell_id)
        # Get the source pixel id
        for cell_id in diff_nbs_cell_id.unique():
            if cell_id == 0:
                # Unselected
                continue
            
            # For each cell id, replace the cell id with the corresponding cell_type value from source_pixels (that is, the channel 1 from where the cell_id is the one selected, using a max operation to aggregate)
            cell_type_of_this_id = source_pixels[1, source_pixels[0]==cell_id].max()
            diff_nbs_cell_type = torch.where(diff_nbs_cell_id == cell_id, cell_type_of_this_id, diff_nbs_cell_type)

        # Now replace each cell_type of the neigbors with the adhesion coefficient [src_type, trg_type]

        for src_type in source_pixels[1].unique():
            # We want to get the neighbors where:
            # - corresponds to a given source pixel (in x,y)
            # - have a different value from the source pixel
            # - have a given target pixel value
            
            # Spatially select only the current src_type
            source_mask = (source_pixels[1] == src_type)
            
            for trg_type in diff_nbs_cell_type.unique():
                if trg_type == 0 or src_type == 0:
                    # Unselected pixels
                    continue

                # Map the background (-1) to first index of the adhesion matrix
                adh_coeff = adhesion_matrix[max(src_type, 0), max(trg_type, 0)]
                
                # FIXME: It is correct to pick the first dimension here?
                masked_neighbors = source_mask.unsqueeze(0) & (diff_nbs_cell_type == trg_type)
                adhesion_coeffs = adhesion_coeffs.where(masked_neighbors, adh_coeff)
                

        return adhesion_coeffs.sum(axis=0)

    def __call__(self,  current_state: torch.Tensor, 
                        current_state_nbs: torch.Tensor, 
                        predicted_state: torch.Tensor,
                        predicted_state_nbs: torch.Tensor):
        """
            Calculates the energy difference of the system if the source pixels are changed to the target pixels.

            Args:
                source_pixels (torch.Tensor): Tensor of shape [h, w] containing the source pixels values or 0 if the pixel is not selected.
                chosen_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors to copy the source pixels to.
                current_diff_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors values of each pixel (restricted to only neighbors that are different from the source pixel).
                adhesion_matrix (torch.Tensor): Tensor of shape [n, n] containing the adhesion values between the different cell types.
                neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors values of each pixel.
                lambda_adhesion (float): Weight of the adhesion energy in the total
        """
        adhesion_matrix = self.config.adhesion_matrix
        current_energy = self.h(source_pixels=current_state, current_neighbors=current_state_nbs, adhesion_matrix=adhesion_matrix)
        predicted_energy = self.h(source_pixels=predicted_state, current_neighbors=predicted_state_nbs, adhesion_matrix=adhesion_matrix)
        return predicted_energy - current_energy


class TorchCPMVolumeConstraint(TorchCPMConstrint):
    """
        Energy coefficient that represent the tendency of different cell types to keep a given volume.
    """

    def __init__(self, cpm_config: TorchCPMConfig):
        super().__init__(cpm_config)
    

    def h(self, volumes, preferred_volumes, lambda_volume=1.0):
        """
            Energy function that penalizes the difference between the current volume and the preferred volume
        """
        v_diff = preferred_volumes - volumes
        return lambda_volume*(v_diff**2)

    def __call__(self, current_state: torch.Tensor, current_selected_neighbors: torch.Tensor, predicted_state: torch.Tensor, predicted_diff_nbs: torch.Tensor):
        """
            Calculates the energy difference of the system if the source pixels are changed to the target pixels.

            Args:
                source_pixels (torch.Tensor): Tensor of shape [h, w] containing the source pixels values or 0 if the pixel is not selected.
                chosen_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors to copy the source pixels to.
                current_diff_neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors values of each pixel (restricted to only neighbors that are different from the source pixel).
                adhesion_matrix (torch.Tensor): Tensor of shape [n, n] containing the adhesion values between the different cell types.
                neighbors (torch.Tensor): Tensor of shape [8, h, w] containing the neighbors values of each pixel.
                lambda_adhesion (float): Weight of the adhesion energy in the total
        """
        # Add the background to the preferred volumes
        preferred_volumes = torch.concat([torch.zeros([1]), self.config.preferred_volumes], axis=0)
        lambda_volume = self.config.lambda_volume

        current_volumes = get_volume_map(current_state)
        predicted_volumes = get_volume_map(predicted_state)
        # Pass the first channel of the volume map (class_id) to the the preferred volumes to.
        preferred_current_volumes = map_value_to_objects(current_state[1].int(), preferred_volumes)
        preferred_predicted_volumes = map_value_to_objects(predicted_state[1].int(), preferred_volumes)
        current_energy = self.h(volumes=current_volumes, preferred_volumes=preferred_current_volumes, lambda_volume=lambda_volume)
        predicted_energy = self.h(volumes=predicted_volumes, preferred_volumes=preferred_predicted_volumes, lambda_volume=lambda_volume)
        
        return predicted_energy - current_energy

class TorchCPMLocalPerimeterConstraint(TorchCPMConstrint):
    """
        Energy coefficient that represent the tendency of different cell membranes to keep a given perimeter.
    """

    def __init__(self, cpm_config: TorchCPMConfig):
        super().__init__(cpm_config)

    def h(self, current_perimeters: torch.Tensor, preferred_perimeters: torch.Tensor, lambda_perimeter=1.0):
        """
            Function that calculate the energy of the system based on the perimeter of each object.
        """
        return lambda_perimeter * (preferred_perimeters - current_perimeters)**2
    

    def __call__(self, 
                 current_state: torch.Tensor, 
                 current_diff_nbs: torch.Tensor, 
                 predicted_state: torch.Tensor, 
                 predicted_diff_nbs: torch.Tensor):
        
        preferred_perimeters = self.config.preferred_local_perimeters
        preferred_perimeters = torch.concat([torch.zeros([1]), preferred_perimeters], axis=0).int()

        # Perimeter contribution of each pixel
        curr_local_perimeter = current_diff_nbs.sum(dim=0)
        # Perimeter contribution of each pixel
        pred_local_perimeter = predicted_diff_nbs.sum(dim=0)
        
        preferred_source_perimeters = map_value_to_objects(current_state[1].int(), preferred_perimeters)
        preferred_target_perimeters = map_value_to_objects(predicted_state[1].int(), preferred_perimeters)
        return self.h(curr_local_perimeter, preferred_source_perimeters) - self.h(pred_local_perimeter, preferred_target_perimeters)
