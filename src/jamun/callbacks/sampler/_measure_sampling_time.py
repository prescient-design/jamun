import time
import logging

import torch
import lightning.pytorch as pl


class MeasureSamplingTimeCallback(pl.Callback):
    """
    PyTorch Lightning callback to measure and log sampling time.
    
    This callback tracks:
    - Total sampling time
    - Number of graphs sampled
    - Time per sample
    - Time per batch
    
    The metrics are logged both during and after sampling.
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        """Reset all time tracking metrics."""
        self.total_sampling_time = 0.0
        self.total_num_graphs = 0
        self.batch_times = []
        self.batch_graph_counts = []
        self.start_time = None
        
    def on_sample_start(self, sampler, pl_module):
        """Called when sampling starts."""
        self.reset()
        
    def on_before_sample_batch(self, sampler, batch_idx, **kwargs):
        """Mark the start time before sampling a batch."""
        self.start_time = time.perf_counter()
        
    def on_after_sample_batch(self, sampler, sample, batch_idx, **kwargs):
        """
        Calculate and log sampling time after a batch is sampled.
        
        Args:
            sampler: The sampler object
            sample: List of sample dictionaries
            batch_idx: Current batch index
        """
        if self.start_time is None:
            return
            
        # Calculate time elapsed
        end_time = time.perf_counter()
        time_elapsed = end_time - self.start_time
        
        # Count the number of graphs in this batch
        num_graphs = sum(sample["xhat_traj"].shape[1] for sample in sample)
        
        # Update totals
        self.total_sampling_time += time_elapsed
        self.total_num_graphs += num_graphs
        
        # Store batch information
        self.batch_times.append(time_elapsed)
        self.batch_graph_counts.append(num_graphs)
        
        # Log metrics
        sampler = sampler.fabric
        sampler.log("sampler/batch_sampling_time", time_elapsed)
        sampler.log("sampler/batch_num_graphs", num_graphs)
        sampler.log("sampler/batch_time_per_graph", time_elapsed / num_graphs)
        sampler.log("sampler/total_sampling_time", self.total_sampling_time)
        sampler.log("sampler/total_num_graphs", self.total_num_graphs)
        sampler.log("sampler/avg_time_per_graph", self.total_sampling_time / self.total_num_graphs)
        
        # Log to console
        py_logger = logging.getLogger("jamun")
        py_logger.info(
            f"Sampled batch {batch_idx} with {num_graphs} samples in {time_elapsed:.4f} seconds "
            f"({time_elapsed / num_graphs:.4f} seconds per sample)."
        )
        
        # Reset start time
        self.start_time = None
        
    def on_sample_end(self, sampler, pl_module):
        """Log final statistics when sampling is complete."""
        if self.total_num_graphs == 0:
            return
            
        avg_time = self.total_sampling_time / self.total_num_graphs
        
        # Log final metrics.
        sampler.log("sampler/final_total_sampling_time", self.total_sampling_time)
        sampler.log("sampler/final_total_num_graphs", self.total_num_graphs)
        sampler.log("sampler/final_avg_time_per_graph", avg_time)
        
        # Calculate additional statistics.
        if self.batch_times:
            sampler.log("sampler/min_batch_time", min(self.batch_times))
            sampler.log("sampler/max_batch_time", max(self.batch_times))
            sampler.log("sampler/avg_batch_time", sum(self.batch_times) / len(self.batch_times))
            sampler.log("sampler/std_batch_time", torch.std(torch.tensor(self.batch_times)).item())
        
        # Log to console.
        py_logger = logging.getLogger("jamun")
        py_logger.info(
            f"Total sampling time: {self.total_sampling_time:.4f} seconds "
            f"for {self.total_num_graphs} samples "
            f"({avg_time:.4f} seconds per sample)."
        )