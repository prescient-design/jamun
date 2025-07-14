import torch
from jamun.data._mdtraj import MDtrajDataset


class RepeatedPositionDataset(MDtrajDataset):
    """
    Dataset that replaces hidden states with copies of the current position.
    This is used for Model 3 experiment where the structures passed to the denoiser
    are copies of the same structure given by y.pos. The denoiser will add noise during training.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize but store total_lag_time before modifying parent behavior."""
        # Store the total_lag_time for our own use
        self._target_total_lag_time = kwargs.get('total_lag_time', 2)
        
        # Prevent parent from doing lag processing by removing lag parameters
        kwargs_no_lag = kwargs.copy()
        kwargs_no_lag['total_lag_time'] = None
        kwargs_no_lag['lag_subsample_rate'] = None
        
        super().__init__(*args, **kwargs_no_lag)

    def __getitem__(self, idx):
        """Override to create position copies instead of using real hidden states."""
        # Get the normal item from parent class (without lag processing)
        graph = super().__getitem__(idx)
        
        # Create the number of hidden states we want based on our target total_lag_time
        num_hidden_states = self._target_total_lag_time - 1
            
        graph.hidden_state = []
        for _ in range(num_hidden_states):
            # Create a copy of the current position (no noise added here)
            graph.hidden_state.append(graph.pos.clone())
            
        return graph 