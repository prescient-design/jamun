# KALA-JAMUN: Spatiotemporal Conditional Generation Documentation

## Introduction

KALA-JAMUN introduces conditioning into the JAMUN workflow to enable temporal-aware molecular generation. The key innovation is conditioning the denoising process on past noisy states, allowing the model to learn and maintain temporal correlations in molecular dynamics. This enhancement necessitated significant changes across three core components of the system:

1. **Modified Dataset Infrastructure**: To support conditioning on historical states
2. **Enhanced Model Architectures**: To process both current and historical information
3. **Memory-Aware Sampling**: To maintain temporal consistency during generation

The conditioning mechanism works by feeding past noisy states directly to the model as part of the input data. This enables the model to learn temporal dependencies and generate more realistic molecular trajectories that respect the underlying dynamics.

This document provides a comprehensive guide covering the complete KALA-JAMUN workflow from data preparation through model architecture to sampling procedures.

## Table of Contents

1. [Chapter 1: Datasets](#chapter-1-datasets)
2. [Chapter 2: Architecture](#chapter-2-architecture)
3. [Chapter 3: Sampling](#chapter-3-sampling)

---

## Chapter 1: Datasets

### Overview

In KALA-JAMUN, the conditioning is based on past noisy states that are fed directly to the model as part of the input data. This required fundamental modifications to the data structure itself.

### Data Structure Modifications

The core data class `DataWithResidueInformation` has been enhanced with a new field to support temporal conditioning:

**Source:** [`src/jamun/utils/data_with_residue_info.py`](src/jamun/utils/data_with_residue_info.py), lines 5-16

```python
class DataWithResidueInformation(torch_geometric.data.Data):
    """Graph with residue-level information."""
    
    pos: torch.Tensor
    atom_type_index: torch.Tensor
    atom_code_index: torch.Tensor
    residue_code_index: torch.Tensor
    residue_sequence_index: torch.Tensor
    residue_index: torch.Tensor
    num_residues: int
    loss_weight: float
    hidden_state: Any  # NEW: Stores past trajectory states
```

**Key Addition:**
- **`hidden_state`**: This new field stores the historical molecular configurations that enable temporal conditioning. It keeps the past trajectories that the model will condition on during the denoising process.

This modification allows the data graph to carry both current molecular state (`pos`) and historical context (`hidden_state`), enabling the model to learn temporal dependencies in molecular dynamics.

### 1.1 MDTrajDataset with Subsampling

The core dataset class `MDTrajDataset` has been enhanced to support KALA-JAMUN's conditioning mechanism through historical state management.

#### Enhanced MDTrajDataset Structure

When a data graph is processed in KALA-JAMUN:
- **`graph.pos`**: Contains the current molecular state (positions)
- **`graph.hidden_state`**: Contains `total_lag_time - 1` past states

The historical states are selected using two key parameters:
- **`lag_subsample_rate`**: Temporal difference between consecutive stored states
- **`total_lag_time`**: Total number of states stored (including the present state)

#### Subsampling Implementation

**Source:** [`src/jamun/data/_mdtraj.py`](src/jamun/data/_mdtraj.py), lines 249-261 (in MDTrajDataset.__init__)

```python
if total_lag_time is not None and lag_subsample_rate is not None:
    lagged_indices = get_subsampled_indices(
        self.traj.n_frames, subsample, total_lag_time, lag_subsample_rate
    )
    # Extract subsampled indices (first element of each list)
    subsampled_indices = [indices[0] for indices in lagged_indices]
    # Extract lagged indices (all except first element)
    self.lagged_indices = [indices[1:] for indices in lagged_indices]
    # Subsample the trajectory using the subsampled indices
    self.hidden_state = [self.traj[indices] for indices in self.lagged_indices]
    self.traj = self.traj[subsampled_indices] # self.traj is permanently modified.
```

**Example**: With `total_lag_time=5` and `lag_subsample_rate=10`:
- Present state: frame 100
- Historical states: frames 90, 80, 70, 60
- Total stored: 5 states (1 current + 4 historical)

### 1.2 Loading with parse_datasets_from_directory

Such datasets can be loaded using the `parse_datasets_from_directory` function:

**Source:** [`src/jamun/data/_utils.py`](src/jamun/data/_utils.py), lines 38-49 (function definition)

```python
datasets = parse_datasets_from_directory(
    root="/data/trajectories",
    traj_pattern=r"traj_(\w+)\.dcd",
    pdb_pattern=r"(\w+)\.pdb",
    total_lag_time=5,
    lag_subsample_rate=10,
    max_datasets=100
)
```

This function automatically:
- Discovers trajectory files using regex patterns
- Matches trajectory files with corresponding PDB topology files
- Creates `MDTrajDataset` objects with proper historical state management
- Applies the index subsampling procedure to populate `hidden_state`

### 1.3 RepeatedPositionDataset (Multimeasurement)

Multimeasurement refers to collecting T independent noisy copies of the same present state. This enables the model to learn from multiple noise realizations applied to identical molecular configurations, improving robustness and sample diversity.

#### Concept

Instead of using historical states from different time points, multimeasurement uses:
- **`batch.pos`**: The current molecular state
- **`batch.hidden_state`**: Contains `T-1` copies of the same `pos`

When noise is independently added in the denoiser, independent realizations of the noise get added to the exact same underlying state, allowing the model to learn the noise distribution more effectively.

#### MDTrajRepeatedDataset

To enable multimeasurement, KALA-JAMUN provides a specialized dataset:

**Source:** [`src/jamun/data/noisy_position_dataset.py`](src/jamun/data/noisy_position_dataset.py), lines 5-37

```python
class RepeatedPositionDataset(MDtrajDataset):
    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        graph = super().__getitem__(idx)
        # Hidden state contains repeated copies of the current position
        # instead of historical states from different time points
        return graph
```

#### Loading with parse_repeated_position_datasets_from_directory

Multimeasurement datasets are created using:

**Source:** [`src/jamun/data/_utils.py`](src/jamun/data/_utils.py), lines 362-373 (function definition)

```python
datasets = parse_repeated_position_datasets_from_directory(
    root="/data/trajectories", 
    traj_pattern=r"traj_(\w+)\.dcd",
    pdb_pattern=r"(\w+)\.pdb",
    # No temporal parameters - using repeated states instead
    max_datasets=100
)
```

**Key Difference**: 
- **Temporal Conditioning**: `hidden_state` contains past states from different time points
- **Multimeasurement**: `hidden_state` contains repeated copies of the current state

This approach allows the model to:
1. Learn from multiple noise realizations on the same state
2. Improve denoising performance through ensemble-like training
3. Generate more diverse samples from the same initial condition

---

## Chapter 2: Architecture

### Overview

KALA-JAMUN implements a new denoiser class, `denoiser_conditional.Denoiser`, which handles the internal training process for temporal conditioning. This model consists of two main operating submodules, the conditioning module and the architecture module, called according to the following workflow: 

1. Just like `jamun.model.Denoiser`, the modules within `jamun.model.denoiser_conditional.Denoiser` are called from within `xhat_normalized`. 
2. The **Conditioning Module**: `denoiser_conditional.Denoiser.conditioner` - first calculates features based on historical states
3. Next, the **Architecture Module**: `denoiser_conditional.Denoiser.g` - processes features from the conditioning module into the final output, which is combined with the noisy coordinates with the normalization factors to construct xhat.


**Key Components:**
1. **Conditioning Module**: Implements various conditioning strategies, with the spatiotemporal conditioner being the most sophisticated
2. **Architecture Module (model.g)**: Enhanced E3Conv variants that handle conditional inputs
3. **Training Process**: Integrated mean centering, alignment, scaling, propagation, and loss computation

**⚠️ Important**: The input signatures for `denoiser_conditional.Denoiser` do not match those of the original `Denoiser`. 



### 2.1 Conditioning Module

Every conditioning module in KALA-JAMUN is of the class `Conditioner`, which defines the interface for calculating features based on historical states.

**Source:** [`src/jamun/model/conditioners/conditioners.py`](src/jamun/model/conditioners/conditioners.py)

#### Available Conditioning Modules

KALA-JAMUN provides several conditioning strategies:

1. **PositionConditioner**: Returns just the input positions (baseline)
2. **MeanConditioner**: Provides mean-centered positions and repeated structures  
3. **SpatioTemporalConditioner**: Uses spatiotemporal processing for feature extraction (most sophisticated)

#### SpatioTemporalConditioner

The most sophisticated conditioning module uses a spatiotemporal GNN to output features based on both current and historical states. This conditioner processes the input through a complete spatiotemporal architecture before passing features to the main denoiser.

**Source:** [`src/jamun/model/conditioners/conditioners.py`](src/jamun/model/conditioners/conditioners.py), SpatioTemporalConditioner class

```python
class SpatioTemporalConditioner(pl.LightningModule):
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        # Process through spatiotemporal model
        spatial_features = self.spatiotemporal_model(y, c_noise=self.c_noise)
        
        # Return [positions, features] for concatenation
        return [y.pos, spatial_features]
```

### 2.2 Spatiotemporal Model (E3SpatioTemporal)

The spatiotemporal model is the core of the `SpatioTemporalConditioner`. It processes molecular data through several stages:

**Source:** [`src/jamun/model/arch/spatiotemporal.py`](src/jamun/model/arch/spatiotemporal.py), lines 403+

#### Architecture Components

1. **Spatial Module**: Processes individual molecular graphs (current and historical states)
2. **Temporal Graph Construction**: Converts spatial graphs into temporal graphs with temporal connections
3. **Temporal Module (E3Transformer)**: Applies transformer architecture on temporal graphs
4. **Reconversion**: Converts temporal graph features back to spatial representation

#### Spatial Module Processing

The spatial module processes each molecular configuration (current and historical) independently:

```python
# Process current positions
node_attr_current = self.spatial_module(
    pos=batch.pos, 
    topology=topology, 
    batch=batch.batch,
    num_graphs=batch.num_graphs,
    c_noise=c_noise,
    effective_radial_cutoff=self.radial_cutoff
)

# Process historical positions
for hidden_pos in batch.hidden_state:
    node_attr_hidden = self.spatial_module(
        pos=hidden_pos,
        topology=topology,
        batch=batch.batch, 
        num_graphs=batch.num_graphs,
        c_noise=c_noise,
        effective_radial_cutoff=self.radial_cutoff
    )
```

#### Temporal Graph Construction

After spatial processing, the individual graphs are converted into temporal graphs where nodes across different time steps are connected:

**Source:** [`src/jamun/model/arch/spatiotemporal.py`](src/jamun/model/arch/spatiotemporal.py), create_temporal_graph function

The temporal graph construction creates edges between the intertemporal copies of the same atom. This gives some freedom as to how to define the connectivity structure of the temporal graph. Three stratgies that have been explored are: 

1. Fan graph--the present node connects to all nodes, and the ith historical node connects to the (i+1)th and (i-1)th historical node (whenever such nodes are available). 
2. Hub and spoke--the present node connects to all nodes, and no historical nodes are mutually connected. 
3. Complete graph--all nodes are mutually connected.

In the temporal graph we also need to define what features the nodes and the edges have. This will be discussed below. 

#### E3Transformer (Temporal Module)

The temporal module applies a transformer architecture specifically designed for temporal graphs:

**Source:** [`src/jamun/model/arch/spatiotemporal.py`](src/jamun/model/arch/spatiotemporal.py), lines 217-284

##### Temporal Embeddings and Encoding Functions

The E3Transformer uses several specialized encoding functions to handle temporal information:

**Key Parameters:**
- **`irreps_node_attr_temporal`**: Irreducible representations for temporal node attributes (default: "1x1e")
- **`node_attr_temporal_encoding_function`**: Encodes temporal position information (default: "gaussian")  
- **`edge_attr_temporal_encoding_function`**: Encodes temporal edge attributes (default: "gaussian")
- **`radial_edge_attr_encoding_function`**: Encodes radial distances (default: "gaussian")

##### Temporal Attribute Processing

```python
# Split edge attribute dimensions: radial and temporal
self.radial_edge_attr_dim = self.edge_attr_dim // 2
self.temporal_edge_attr_dim = self.edge_attr_dim - self.radial_edge_attr_dim

# Temporal gate for combining node attributes with temporal position
irreps_with_temporal = self.irreps_node_attr + self.irreps_node_attr_temporal
self.temporal_gate = e3tools.nn.GateWrapper(
    irreps_in=irreps_with_temporal,
    irreps_out=self.irreps_hidden,
    irreps_gate=irreps_with_temporal,
)
```

The temporal gate combines:
- **Node attributes**: From spatial processing of individual timesteps  
- **Temporal position**: Encoded position in the temporal sequence
- **Temporal edges**: Connections between atoms across different timesteps

##### Transformer Layers

The temporal transformer processes the combined spatial-temporal information through multiple attention layers:

**Source:** [`src/jamun/model/arch/spatiotemporal.py`](src/jamun/model/arch/spatiotemporal.py), lines 267-284

```python
for _ in range(num_layers):
    self.layers.append(
        e3tools.nn.TransformerBlock(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            edge_attr_dim=self.edge_attr_dim,
            num_heads=self.num_attention_heads,
            conv=self.conv,
        )
    )
```

#### Integrating Pretrained Spatial Modules

A powerful feature of KALA-JAMUN is the ability to use a pretrained unconditional JAMUN model as the spatial module within the spatiotemporal architecture. This enables leveraging existing trained models as building blocks for more sophisticated temporal conditioning.

##### Architecture Overview

In this setup, we have:
- **Overlying Conditional Denoiser**: `jamun.model.denoiser_conditional.Denoiser` - the main KALA-JAMUN model
- **Sub-Denoiser**: `jamun.model.Denoiser` - the pretrained unconditional JAMUN model used as spatial module

The overlying conditional denoiser has its own normalization factor `c_in`, but when scaled data `y_scaled = c_in * y` goes into the sub-denoiser, this scaling must be divided out since the sub-denoiser has its own internal normalization.

##### DenoiserWrapper: Input Signature Unification

The core challenge is that `Denoiser.xhat` and `E3Conv` have different input signatures, requiring a wrapper to make them compatible:

**Source:** [`src/jamun/utils/pretrained_wrapper.py`](src/jamun/utils/pretrained_wrapper.py), lines 56-85

```python
class DenoiserWrapper(nn.Module):
    """
    Wrapper around a denoiser model that matches the spatial module interface.
    
    This allows pretrained denoiser models to be used as spatial/temporal modules
    in the spatiotemporal architecture by replicating the full denoiser logic
    including normalization factors computed from the denoiser's own parameters.
    """
    
    def __init__(self, denoiser_model: nn.Module, c_in: float = 1.0, trainable: bool = True):
        """
        Args:
            denoiser_model: The pretrained denoiser model
            c_in: Rescaling factor to convert positions from overlaying model scale
            trainable: Whether to keep the model trainable (default: True)
        """
        super().__init__()
        self.denoiser = denoiser_model
        self.c_in = c_in  # Rescaling factor from overlying denoiser
```

##### Rescaling Mechanism

The `DenoiserWrapper` handles the critical `c_in` rescaling between the overlying and sub-denoiser:

**Source:** [`src/jamun/utils/pretrained_wrapper.py`](src/jamun/utils/pretrained_wrapper.py), lines 98-118

```python
def forward(self, pos, topology, batch, num_graphs, c_noise, effective_radial_cutoff):
    # Sample sigma from the denoiser's own sigma distribution
    sigma = self.denoiser.sigma_distribution.sample().to(pos.device)
    
    # CRITICAL: Rescale positions from overlaying model scale
    y = pos / self.c_in  # Divide out overlying denoiser's c_in
    
    # Apply sub-denoiser's own normalization
    c_in, c_skip, c_out, _ = compute_normalization_factors(
        sigma,
        average_squared_distance=self.denoiser.average_squared_distance,
        normalization_type=self.denoiser.normalization_type,
        sigma_data=self.denoiser.sigma_data,
        D=y.shape[-1],
        device=y.device,
    )
```

**Key Steps:**
1. **Input Rescaling**: `y = pos / self.c_in` - divides out the overlying denoiser's scaling
2. **Internal Normalization**: The sub-denoiser computes its own `c_in`, `c_skip`, `c_out` 
3. **Denoiser Processing**: Full `xhat_normalized` logic is replicated internally
4. **Output**: Features compatible with the spatiotemporal architecture

This design allows seamless integration of pretrained models while maintaining proper normalization hierarchies between the overlying conditional denoiser and the embedded unconditional denoiser.

### 2.3 Architecture Module (model.g)

The architecture module `model.g` is the main processing unit that receives features from the conditioning module. Unlike standard E3Conv models, these variants are designed to handle conditional inputs.

#### E3ConvConditional vs Standard E3Conv

The key difference from standard E3Conv models is the ability to process multiple input structures and conditional information:

**Source:** [`src/jamun/model/arch/e3conv_conditional.py`](src/jamun/model/arch/e3conv_conditional.py), lines 15-40

```python
class E3ConvConditional(torch.nn.Module):
    def __init__(
        self,
        # Standard E3Conv parameters...
        N_structures: int = 1,  # NEW: Number of input structures
        # ...
    ):
```

**Key Features:**
- **Multi-Structure Support**: Processes multiple molecular structures simultaneously via `N_structures`
- **Noise Conditioning**: Integrates noise level information throughout the network
- **Skip Connections**: Uses noise-conditional skip connections for stable training

#### E3ConvConditionalSpatioTemporal

A specialized variant designed specifically for spatiotemporal conditioning:

**Source:** [`src/jamun/model/arch/e3conv_conditional.py`](src/jamun/model/arch/e3conv_conditional.py), lines 312+

This variant handles concatenated position and feature data from the spatiotemporal model:

```python
def forward(
    self,
    pos: Tensor,  # [N, 3 + spatial_features_dim] from [y.pos, spatial_features]
    topology: torch_geometric.data.Batch,
    c_noise: Tensor,
    effective_radial_cutoff: float,
) -> Tensor:
    # Split positions: first 3 coords are physical, rest are features
    pos_physical = pos[:, :3]  # [N, 3] - physical coordinates
    pos_features = pos[:, 3:]  # [N, spatial_features_dim] - spatial features
    
    # Compute edge attributes using ONLY physical positions
    edge_vec_physical = pos_physical[src] - pos_physical[dst]
    edge_sh = self.sh(edge_vec_physical)
    
    # Combine node attributes with spatial features
    combined_attr = torch.cat([node_attr, pos_features], dim=-1)
    node_attr = self.spatial_feature_aggregator(combined_attr)
```

**Design Principle**: Separates physical coordinates (for geometric operations) from feature coordinates (for conditioning).

### 2.4 Noising and Denoising Process

KALA-JAMUN implements a comprehensive training pipeline that handles mean centering, alignment, scaling, model propagation, and loss computation.

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), xhat and xhat_normalized methods

#### Complete Denoising Workflow

The denoising process follows these steps:

##### 1. Mean Centering (Input Preparation)

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), lines 308-311

```python
if self.mean_center:
    y = mean_center(y)
    y = self._mean_center_hidden_states(y)
```

Both current and historical states are mean-centered to ensure translational invariance.

##### 2. Alignment (if enabled)

Molecular configurations are aligned to a reference to handle rotational variance. This is done after adding noise but before denoising during training.

##### 3. Scaling (Normalization)

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), lines 264-286

```python
# Compute normalization factors
c_in, c_skip, c_out, c_noise = self.normalization_factors(sigma, D)

# Scale input positions and hidden states
y_scaled = y.clone()
y_scaled.pos = y.pos * c_in
if hasattr(y, "hidden_state") and y.hidden_state is not None:
    y_scaled.hidden_state = []
    for positions in y.hidden_state:
        y_scaled.hidden_state.append(positions * c_in)
```

**Key scaling factors:**
- **`c_in`**: Scales input coordinates and hidden states
- **`c_skip`**: Skip connection scaling  
- **`c_out`**: Output scaling
- **`c_noise`**: Noise conditioning scaling

##### 4. Model Propagation

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), lines 294-299

```python
# Step 1: Conditioning
conditioned_structures = self.conditioner(y_scaled)

# Step 2: Architecture processing
g_pred = self.g(torch.cat([*conditioned_structures], dim=-1), 
                topology=y_scaled, 
                c_noise=c_noise, 
                effective_radial_cutoff=radial_cutoff)
```

The conditioner processes scaled inputs and hidden states, then the architecture module processes the concatenated conditioned structures.

##### 5. Skip Connection and Output Scaling

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), lines 301-304

```python
# Apply skip connection and output scaling
xhat.pos = c_skip * y.pos + c_out * g_pred

# Update hidden state for next iteration
if hasattr(y, "hidden_state") and y.hidden_state is not None:
    xhat.hidden_state = [y.pos, *y.hidden_state[:-1]]
```

##### 6. Mean Centering (Output)

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), lines 317-319

```python
# Mean center the prediction
if self.mean_center:
    xhat = mean_center(xhat)
```

##### 7. Loss Computation

The loss is computed between the denoised prediction and the clean target, typically using MSE loss on the positions while maintaining proper handling of the hidden states for temporal consistency.

#### Key Differences from Standard Denoiser

1. **Hidden State Handling**: All scaling operations are applied to both current and hidden states
2. **Conditioning Integration**: The conditioner processes scaled inputs before the main architecture
3. **Temporal Consistency**: Hidden states are properly updated to maintain temporal sequence
4. **Multi-Structure Processing**: Conditioned structures are concatenated before processing

This comprehensive pipeline ensures that KALA-JAMUN properly handles temporal information throughout the entire denoising process, maintaining consistency between current and historical states while applying the appropriate transformations for effective learning.

---

## Chapter 3: Sampling

### Overview

Once the KALA-JAMUN model is trained, it is time to sample using the score function obtained via the Miyasawa-Tweedie formula from the denoiser. The score function relates the denoised prediction to the true score of the data distribution:

```
score(y, σ) = (x̂(y, σ) - y) / σ²
```

Where `x̂(y, σ)` is the denoised prediction from the trained model.

Since KALA-JAMUN conditions on historical states, the sampling process must be modified to properly handle memory. This requires changes to the standard ABOBA and BAOAB samplers to account for the temporal dependencies and ensure that historical information is correctly propagated through the sampling chain.

### 3.1 Model Loading

Loading the trained conditional model is straightforward:

**Source:** [`src/jamun/model/denoiser_conditional.py`](src/jamun/model/denoiser_conditional.py), load_from_checkpoint method

```python
model = denoiser_conditional.Denoiser.load_from_checkpoint(checkpoint_path)
```

The loaded model retains its conditional structure with both the conditioning module and architecture module properly initialized.

### 3.2 Memory-Aware Sampling Modifications

#### Changes to ABOBA and BAOAB Samplers

The standard ABOBA and BAOAB sampling algorithms must be modified to handle historical states. The key changes involve:

1. **State Representation**: Instead of just current positions `y`, we now track `(y, y_hist)` where `y_hist` is a list of historical states
2. **Score Function Interface**: The score function now takes both current and historical states as input
3. **Memory Updates**: Historical states must be updated periodically during sampling to maintain temporal consistency

#### Modified BAOAB with Memory

The memory-aware BAOAB sampler uses a two-loop structure to handle temporal conditioning:

**Source:** [`src/jamun/sampling/mcmc/functional/_splitting.py`](src/jamun/sampling/mcmc/functional/_splitting.py), lines 255-327

```python
def baoab_memory(
    y: torch.Tensor,              # Current positions
    y_hist: list,                 # Historical states list
    score_fn: Callable,           # Score function accepting (y, y_hist)
    steps: int,
    history_update_frequency=1,   # Inner loop length
    **kwargs
):
    """BAOAB splitting scheme with two-loop structure for memory updates."""
    
    # Initialize velocity and score processing
    v = initialize_velocity(v_init=v_init, y=y, u=u)
    score_fn_processed = create_score_fn(score_fn, inverse_temperature, score_fn_clip)
    psi, orig_score = score_fn_processed(y, y_hist=y_hist)
    
    # OUTER LOOP: Iterate over memory updates
    for i in range(1, steps):
        
        # INNER LOOP: Equilibrate to conditional density p(y_t | y_hist)
        for j in range(1, history_update_frequency):
            y_current = y.clone().detach()
            
            # Standard BAOAB steps with FIXED history
            v = v + u * (delta / 2) * psi          # B: velocity update
            y = y + (delta / 2) * v                # A: position update  
            R = torch.randn_like(y)
            vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R  # O: Ornstein-Uhlenbeck
            y = y + (delta / 2) * vhat             # A: position update
            psi, orig_score = score_fn_processed(y, y_hist=y_hist)  # B: score update
            v = vhat + (delta / 2) * psi
        
        # MEMORY UPDATE: Shift history after equilibration
        y_hist.pop(-1)              # Remove oldest state
        y_hist.insert(0, y_current) # Add equilibrated state to history
```

##### Two-Loop Structure and Conditional Equilibration

**Outer Loop (Memory Updates):**
- Iterates `steps` times over the complete sampling process
- Each iteration updates the historical memory `y_hist`
- Represents the temporal progression of the molecular system

**Inner Loop (Conditional Equilibration):**
- Runs for `history_update_frequency` steps with **fixed historical context**
- Equilibrates the current state `y` to the conditional density `p(y_t | y_hist)`
- The historical states `y_hist` remain constant during this equilibration

**Conditional Density Equilibration:**
The inner loop is crucial because it allows the sampler to properly explore the conditional distribution given the current historical context. Without sufficient equilibration:
- The sampler might not fully explore `p(y_t | y_hist)` before updating history
- This could lead to poor mixing and biased samples
- The temporal correlations learned during training might not be properly respected

**Key Parameters:**
- **`history_update_frequency`**: Controls the balance between:
  - **Computational cost**: Higher values require more inner loop steps
  - **Sampling quality**: Longer equilibration ensures better conditional sampling
  - **Temporal accuracy**: More frequent updates maintain tighter temporal consistency

### 3.3 Score Function Wrapper Modifications

The score function wrapper has been modified to handle current and historical states differently:

**Source:** [`src/jamun/utils/sampling_wrapper.py`](src/jamun/utils/sampling_wrapper.py), lines 132-140

```python
def score(self, y, y_hist, sigma):
    """Score function that handles current and historical states."""
    graph = self.positions_to_graph(y, y_hist).to(self.device)
    return self._model.score(graph, sigma)

def positions_to_graph(self, y, y_hist):
    """Convert positions and history to graph format."""
    graph = self.init_graphs.clone()
    graph.pos = y
    graph.hidden_state = y_hist  # Assign historical states
    return graph
```

**Key Changes:**
1. **Dual Input**: Score function now accepts both `y` (current) and `y_hist` (historical) positions
2. **Graph Construction**: `positions_to_graph` method creates data graphs with `hidden_state` populated from `y_hist`
3. **Model Interface**: The wrapped model's score method processes the complete graph with temporal information

### 3.4 ModelSamplingWrapperMemory

The sampling wrapper provides the interface between the memory-aware samplers and the conditional model:

**Source:** [`src/jamun/utils/sampling_wrapper.py`](src/jamun/utils/sampling_wrapper.py), lines 95-127

```python
class ModelSamplingWrapperMemory:
    """Wrapper for models that depend on a memory of states."""
    
    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float, recenter_on_init: bool = True):
        self._model = model
        self.init_graphs = init_graphs
        self.sigma = sigma
        
        # Mean center both positions and hidden states
        if recenter_on_init:
            self.init_graphs = mean_center(self.init_graphs)
            if hasattr(self.init_graphs, 'hidden_state') and self.init_graphs.hidden_state:
                for i in range(len(self.init_graphs.hidden_state)):
                    mean = scatter(self.init_graphs.hidden_state[i], self.init_graphs.batch, dim=0, reduce="mean")
                    self.init_graphs.hidden_state[i] = self.init_graphs.hidden_state[i] - mean[self.init_graphs.batch]
    
    def sample_initial_noisy_positions(self) -> torch.Tensor:
        """Sample initial noisy current positions."""
        pos = self.init_graphs.pos
        return pos + torch.randn_like(pos) * self.sigma
    
    def sample_initial_noisy_history(self) -> list:
        """Sample initial noisy historical states."""
        noisy_history = []
        for hidden_state in self.init_graphs.hidden_state:
            noisy_history.append(hidden_state + torch.randn_like(hidden_state) * self.sigma)
        return noisy_history
```

**Key Features:**
- **Dual Initialization**: Separately initializes current positions and historical states with noise
- **Mean Centering**: Applies mean centering to both current and historical states
- **Memory Management**: Handles the `hidden_state` list structure properly

### 3.5 SamplerMemory Module

The `SamplerMemory` class wraps the memory-aware sampling functionality:

**Source:** [`src/jamun/sampling/_sampler.py`](src/jamun/sampling/_sampler.py), lines 101-130

```python
class SamplerMemory(Sampler):
    """A sampler for molecular dynamics simulations that uses memory."""
    
    def sample(
        self,   
        model,
        batch_sampler,
        num_batches: int,
        init_graphs: torch_geometric.data.Data,
        continue_chain: bool = False,
    ):
        # Setup model and device
        self.fabric.launch()
        self.fabric.setup(model)
        model.eval()
        
        # Create memory-aware wrapper
        model_wrapped = utils.ModelSamplingWrapperMemory(
            model=model,
            init_graphs=init_graphs,
            sigma=batch_sampler.sigma,
        )
        
        # Initialize with memory
        y_init = model_wrapped.sample_initial_noisy_positions()
        y_hist_init = model_wrapped.sample_initial_noisy_history()
```

**Responsibilities:**
- **Model Wrapping**: Creates `ModelSamplingWrapperMemory` instance
- **Memory Initialization**: Sets up both current and historical initial states
- **Sampling Coordination**: Manages the overall sampling process with memory

### 3.6 Memory Loop Mechanics

The memory loop in KALA-JAMUN sampling works as follows:

1. **Initialization**: 
   - Current positions: `y_init` (noisy version of initial state)
   - Historical states: `y_hist_init` (noisy versions of `hidden_state`)

2. **Sampling Step**: 
   - Compute score using both current and historical states
   - Update current positions using BAOAB/ABOBA dynamics
   - Periodically update historical memory

3. **Memory Update**:
   - Shift historical states: `y_hist = [y_current, y_hist[:-1]]`
   - Maintain constant memory length matching training configuration

4. **Temporal Consistency**:
   - Memory update frequency controls temporal coherence
   - Balance between computational cost and temporal accuracy

This design ensures that the sampling process respects the temporal dependencies learned during training while maintaining computational efficiency through configurable memory update frequencies.

---

## Chapter 4: Usage

### Overview

This chapter describes the experimental setup for KALA-JAMUN, covering the enhanced dataset generation, training procedures, and sampling protocols. The experiments focus on alanine dipeptide (ALA_ALA) systems with enhanced sampling data to evaluate temporal conditioning performance.

### 4.1 Enhanced Sampled Data

#### Dataset Overview

KALA-JAMUN experiments utilize enhanced sampling data from two main series:

1. **ALA_ALA_enhanced**: 5 swarms of 50 frames each from 184 different grid points
2. **ALA_ALA_enhanced_long**: 2 swarms of 100,000 frames each from 184 different grid points

Both datasets consist of swarms sampled at **20 fs intervals**, providing high temporal resolution for learning molecular dynamics.

#### Data Source and Organization

**Source Location**: `/data/bucket/vanib/ALA_ALA/swarms/swarm_results`

The raw swarm data has been reorganized using the script: `scratch/reorganize_swarm_data.py` which sorts the trajectories into training and validation buckets according to different splitting strategies.

#### Data Splitting Strategies

The reorganization script implements four distinct splitting strategies:

##### 1. Grid Split (`grid_split`) 
- **Training Set**: 172 randomly selected grid codes, all trajectories (001-005)
- **Validation Set**: Remaining 12 grid codes, all trajectories
- **Principle**: Complete separation by spatial location in conformational space
- **Use Case**: Tests generalization to unseen regions of conformational space
- **Output**: `/data2/sules/ALA_ALA_enhanced_full_swarm`

##### 2. Trajectory Split (`trajectory_split`)
- **Training Set**: All grid points, trajectories 001-004
- **Validation Set**: All grid points, trajectory 005
- **Principle**: Ensures both train/val cover all conformational regions
- **Use Case**: Tests temporal generalization within known conformational regions
- **Output**: `/data2/sules/ALA_ALA_enhanced_full_grid`

##### 3. Long Grid Split (`long_grid_split`)
- **Training Set**: 172 grid codes, 2000ps trajectories (001, 003)
- **Validation Set**: Remaining 12 grid codes, 2000ps trajectories
- **Principle**: Grid-based splitting with extended trajectories
- **Use Case**: Tests generalization with longer temporal context
- **Output**: `/data2/sules/ALA_ALA_enhanced_long`

##### 4. State Split (`state_split`)
- **Criterion**: Conformational state based on phi/psi angles of first residue
- **Training Set**: Trajectories outside phi ∈ (0,100°), psi ∈ (-50,100°)
- **Validation Set**: Trajectories with first residue in specified phi/psi range
- **Principle**: Complete withholding of specific conformational states
- **Use Case**: Tests ability to generate unseen metastable conformations
- **Output**: `/data2/sules/ALA_ALA_enhanced_long_state_split`

**Script Usage:**
```bash
python reorganize_swarm_data.py SPLITTING_STRATEGY
```

**Available Strategies:**
- `grid_split`: For standard enhanced data with grid-based splitting
- `trajectory_split`: For full grid coverage with trajectory-based splitting  
- `long_grid_split`: For long trajectories with grid-based splitting
- `state_split`: For conformational state-based splitting

#### Using 2000ps Trajectories

For experiments requiring the 2000ps trajectory data (strategies `long_grid_split` and `state_split`), the script automatically uses the longer trajectories. However, if you need to modify this behavior, update the following locations in `reorganize_swarm_data.py`:

**Source:** [`scratch/reorganize_swarm_data.py`](scratch/reorganize_swarm_data.py), lines 468 and 480

```python
# Line 468: In reorganize_with_long_grid_split function
copy_files_for_grid_split(
    SOURCE_DIR,
    os.path.join(target_dir, 'train'),
    train_codes,
    trajectory_codes,
    SINGLE_PDB_FILE,
    'TRAIN',
    use_2000ps=True  # Set to True for 2000ps trajectories
)

# Line 480: In the same function for validation split
copy_files_for_grid_split(
    SOURCE_DIR,
    os.path.join(target_dir, 'val'),
    val_codes,
    trajectory_codes,
    SINGLE_PDB_FILE,
    'VAL',
    use_2000ps=True  # Set to True for 2000ps trajectories
)
```

**Function Parameter:** [`scratch/reorganize_swarm_data.py`](scratch/reorganize_swarm_data.py), line 151

```python
def copy_files_for_grid_split(
    source_dir: str,
    target_dir: str,
    grid_codes: List[str],
    trajectory_codes: List[str],
    single_pdb_file: str,
    split_name: str,
    use_2000ps: bool = False  # Set to True for 2000ps trajectories
):
```

### 4.2 Training

#### Training Configuration

Once data selection is completed, models can be trained using the `train_enhanced_*` configuration series:

**Command Syntax:**
```bash
jamun_train --config-dir=configs experiment={experimental_config_name}
```

### 4.3 Sampling

#### Sampling Configuration

Once training is completed, sampling requires the memory-aware configuration:

**Critical Configuration**: Set `config="sample_memory"` in the sampling script to enable memory-aware sampling with historical state management.

#### Sampling Command

**Standard Syntax:**
```bash
jamun_sample --config-dir=configs experiment={experimental_config_name}
```

**Example:**
```bash
jamun_sample --config-dir=configs experiment=train_enhanced_full_grid
```

#### Memory-Aware Sampling Setup

The `sample_memory` configuration automatically handles:

1. **Model Loading**: Loads conditional denoiser from checkpoint
2. **Memory Initialization**: Sets up initial historical states from validation data
3. **Sampler Selection**: Uses `SamplerMemory` with `baoab_memory` algorithm
4. **Wrapper Configuration**: Employs `ModelSamplingWrapperMemory` for proper interface

### 4.4 Experiments

This section describes key experiments designed to evaluate KALA-JAMUN's performance and validate design choices for temporal conditioning.

#### 4.4.1 Model Comparison

**Objective**: Compare different conditioning strategies and temporal graph topologies to establish the effectiveness of spatiotemporal conditioning.

**Models Compared**:
1. **Standard JAMUN**: Baseline unconditional denoiser without temporal information
2. **Position Conditioner**: Simple conditioning using current positions only
3. **Spatiotemporal Conditioner (Fan Graph)**: Full spatiotemporal model with fan temporal graph topology
4. **Spatiotemporal Conditioner (Hub-and-Spoke)**: Full spatiotemporal model with hub-and-spoke temporal graph topology

For instance, check out this wandb [run](https://genentech.wandb.io/sule-shashank/jamun/runs/scxc4bt4/overview) and its associated group. 


#### 4.4.2 Noise Check (Multimeasurement Validation)

**Objective**: Validate the multimeasurement approach by comparing standard JAMUN with reduced noise against spatiotemporal models using repeated position datasets.

**Experimental Setup**:

**Standard JAMUN Configuration**:
- Noise level: `σ/√T` (reduced noise to account for T measurements)
- Dataset: Standard molecular trajectory data
- Model: Unconditional denoiser

**Spatiotemporal Model Configuration**:
- **Repeated Position Dataset**: `total_lag_time = T` with repeated copies of current state
- **Standard Temporal Dataset**: `total_lag_time = T` with historical trajectory states
- Noise level: Standard `σ`
- Model: Spatiotemporal conditioner

**Experimental Script**: [`scripts/slurm/train_noise_check.sh`](scripts/slurm/train_noise_check.sh)

**Key Comparisons**:
1. **Standard JAMUN (σ/√T)** vs **Spatiotemporal + Repeated Dataset (σ)**
2. **Standard JAMUN (σ/√T)** vs **Spatiotemporal + Temporal Dataset (σ)**
3. **Repeated Dataset** vs **Temporal Dataset** (both with spatiotemporal conditioning)

**Sample wandb run**
Run [here](https://genentech.wandb.io/sule-shashank/jamun/runs/4j8bfj5k/overview) and check out its associated group. 

#### 4.4.3 Total Lag Time vs Lag Subsample Rate Experiment

**Objective**: Systematically evaluate the impact of temporal parameters (`total_lag_time` and `lag_subsample_rate`) across different temporal graph topologies.

**Parameter Space**:
- **Total Lag Time**: Number of historical states included (e.g., 2, 4, 6, 8, 10)
- **Lag Subsample Rate**: Temporal spacing between consecutive states (e.g., 5, 10, 20, 50 timesteps)
- **Graph Types**: Fan, Hub-and-Spoke, Complete graph topologies

**Experimental Design**:
- Grid search across parameter combinations
- Fixed computational budget per configuration
- Consistent evaluation metrics across all runs

**Experimental Script**: [`scripts/slurm/train_graph_type_comparison.sh`](scripts/slurm/train_graph_type_comparison.sh)

**Wandb runs**

Run [here](https://genentech.wandb.io/sule-shashank/jamun/runs/tjwcsf4g/overview) and check out its associated group 
#### 4.4.4 Sampling runs 

1. **Bond degradation** The bond degradation of KALA-JAMUN vs Standard JAMUN with a trajectory of 50K steps was compared. We also compared KALA-JAMUN to a standard JAMUN trained for 500 epochs. The run for KALA jamun is [here](https://genentech.wandb.io/sule-shashank/jamun/runs/1j4us3nx?nw=nwusersuleshashank). The run for Standard JAMUN is [here](https://genentech.wandb.io/sule-shashank/jamun/runs/vigqbemt/overview) and the run for the highly trained standard JAMUN is [here](https://genentech.wandb.io/sule-shashank/jamun/runs/9u4qo5ax/overview). 

2. **Comparing ensembles**: We compared KALA JAMUN vs JAMUN in terms of being able to converge the distribution from the short swarm data (1ps). The results for KALA-JAMUN are [here](https://genentech.wandb.io/sule-shashank/jamun/runs/jwk7i45j/overview) and standard JAMUN are [here](https://genentech.wandb.io/sule-shashank/jamun/runs/u2of58jn/overview). 