# Conditional Generation Branch Documentation

This document provides a comprehensive walkthrough of the conditional-gen branch, covering the spatiotemporal model architecture, conditional denoising, and memory-based sampling mechanisms.

## Table of Contents

1. [Spatiotemporal Model Architecture](#spatiotemporal-model-architecture)
2. [Conditional Denoiser Architecture](#conditional-denoiser-architecture)
3. [Model.g Parameterization and E3Conv Conditional](#modelg-parameterization-and-e3conv-conditional)
4. [Memory-Based Sampling: BAOAB/ABOBA Subroutines](#memory-based-sampling-baoababoba-subroutines)
5. [Sampling Memory Wrapper](#sampling-memory-wrapper)

---

## Spatiotemporal Model Architecture

### Overview

The spatiotemporal model (`E3SpatioTemporal`) implements a complete workflow that processes molecular structures with temporal dependencies by converting between spatial and temporal graph representations.

### Core Components

#### 1. Spatial to Temporal Graph Conversion

The model begins by converting spatial graphs to temporal graphs using the `spatial_to_temporal_graphs()` function:

```python
def spatial_to_temporal_graphs(batch, graph_type="fan"):
    """
    Convert a batch of spatial graphs to temporal graphs with configurable connectivity.
    
    For each spatial node with position + hidden states, create a temporal graph where:
    - Node 0: current position
    - Nodes 1-T: hidden state positions
    - Connectivity depends on graph_type parameter
    """
```

**Graph Type Options:**
- **"fan"**: Hub-spoke + sequential connections (0→all, i→(i+1))
- **"hub_n_spoke"**: Only hub-spoke connections (0→all, no sequential)
- **"complete"**: Complete graph without self-loops (all-to-all excluding self)
- **"complete_no_self"**: Complete graph with self-loops (all-to-all including self)

#### 2. Temporal Position Calculation

Temporal positions are normalized to create consistent temporal embeddings:

```python
def calculate_temporal_positions(temporal_length, mode="linear", device=None):
    """Calculate normalized temporal positions [0, 1/T, 2/T, ..., (T-1)/T]"""
    positions = torch.linspace(0, 1, temporal_length + 1, device=device)[:-1]
    return positions
```

### Complete Spatiotemporal Workflow

The `E3SpatioTemporal.forward()` method implements the following pipeline:

#### Step 1: Spatial Graph → Temporal Graph Conversion
```python
temporal_batch = spatial_to_temporal_graphs(batch, graph_type=self.graph_type)
```

#### Step 2: Spatial Processing of All Time Steps
For each position (current + hidden states), the spatial module processes:
```python
# Current positions
node_attr_current = self.spatial_module(
    pos=batch.pos, 
    topology=topology, 
    batch=batch.batch,
    num_graphs=batch.num_graphs,
    c_noise=c_noise,
    effective_radial_cutoff=self.radial_cutoff
)

# Hidden state positions  
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

#### Step 3: Spatial-Temporal Feature Assembly
```python
node_attr_spatial_temporal = torch.cat(node_attr_list, dim=1)  # [N, T, features]
```

#### Step 4: Spatial→Temporal Pooling
```python
temporal_node_attr = self.spatial_to_temporal_pooler(node_attr_spatial_temporal, temporal_batch)
```

#### Step 5: Temporal Processing
The temporal module processes the temporal graph using an E3 Transformer:
```python
temporal_output = self.temporal_module(
    temporal_node_attr,
    temporal_batch,
    self.radial_cutoff,
    self.temporal_cutoff
)
```

#### Step 6: Temporal→Spatial Pooling
```python
spatial_features = self.temporal_to_spatial_pooler(temporal_output, temporal_batch)
```

#### Step 7: Return to Spatial Representation
The final spatial features are returned for use by the conditional architecture.

---

## Conditional Denoiser Architecture

### Denoiser_Conditional Overview

The `Denoiser` class in `denoiser_conditional.py` extends the standard JAMUN denoiser with conditional generation capabilities through the integration of conditioner modules.

### Key Components

#### 1. Conditioner Integration

The denoiser accepts a `conditioner` parameter that processes the input batch to generate conditioning structures:

```python
def conditioner(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
    if self.conditioning_module is None:
        return self.conditioner_default(y)  # Returns [y.pos]
    elif callable(self.conditioning_module):
        return self.conditioning_module(y)
    else:
        raise ValueError("Conditioner must be a callable or None")
```

#### 2. Hidden State Management

The denoiser properly handles hidden states throughout the pipeline:

**Adding Noise to Hidden States:**
```python
def add_noise(self, x: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]):
    # Add noise to current positions
    y.pos = x.pos + sigma * noise
    # Add noise to hidden states
    for i in range(len(y.hidden_state)):
        y.hidden_state[i] = x.hidden_state[i] + sigma * hidden_noise[i]
```

**Hidden State Alignment:**
```python
def _align_A_to_B_batched_with_hidden_states(self, A: torch_geometric.data.Batch, B: torch_geometric.data.Batch):
    # Align positions
    A_aligned.pos = kabsch_algorithm(A.pos, B.pos, A.batch, A.num_graphs)
    # Align hidden states
    if hasattr(A, "hidden_state") and A.hidden_state is not None:
        A_aligned.hidden_state = []
        for i in range(len(A.hidden_state)):
            A_aligned.hidden_state.append(kabsch_algorithm(
                A.hidden_state[i], B.pos, A.batch, A.num_graphs
            ))
```

### The `xhat_normalized` Method: Core Conditional Processing

The `xhat_normalized` method is the heart of conditional generation:

#### Step 1: Normalization Factor Computation
```python
c_in, c_skip, c_out, c_noise = self.normalization_factors(sigma, D)
radial_cutoff = self.effective_radial_cutoff(sigma) / c_in
```

#### Step 2: Input Scaling
```python
y_scaled = y.clone()
y_scaled.pos = y.pos * c_in
# Scale hidden states
if hasattr(y, "hidden_state") and y.hidden_state is not None:
    y_scaled.hidden_state = []
    for positions in y.hidden_state:
        y_scaled.hidden_state.append(positions * c_in)
```

#### Step 3: Conditioning Structure Generation
```python
with torch.cuda.nvtx.range("conditioning"): 
    conditioned_structures = self.conditioner(y_scaled)
```

The conditioner returns a list of tensors that will be concatenated for model input.

#### Step 4: Model Prediction via model.g
```python
with torch.cuda.nvtx.range("g"):    
    g_pred = self.g(torch.cat([*conditioned_structures], dim=-1), 
                    topology=y_scaled, 
                    c_noise=c_noise, 
                    effective_radial_cutoff=radial_cutoff)
```

**Key Point**: `model.g` receives the concatenated conditioned structures as input positions.

#### Step 5: Output Construction and Hidden State Update
```python
xhat.pos = c_skip * y.pos + c_out * g_pred
if hasattr(y, "hidden_state") and y.hidden_state is not None:
    xhat.hidden_state = [y.pos, *y.hidden_state[:-1]]  # Hidden state shifts forward
```

**Hidden State Evolution**: The hidden state list is updated by:
1. Moving current position (`y.pos`) to the front
2. Keeping all previous hidden states except the oldest one
3. This creates a sliding window of historical positions

---

## Model.g Parameterization and E3Conv Conditional

### Architecture Variants

The conditional-gen branch introduces several E3Conv variants designed for different conditioning scenarios:

#### 1. E3ConvConditional

**Purpose**: Basic conditional model that handles multiple structure inputs
**Key Parameters**:
- `N_structures`: Number of input structures to concatenate
- `irreps_sh`: Extended to `N_structures * self.irreps_sh` for spherical harmonics applied in parallel to N_structures-many 3D vectors. 

**Forward Pass**:
```python
def forward(self, pos: Tensor, topology, c_noise, effective_radial_cutoff):
    # pos should be [batch_size*N, 3T] where T is number of time-steps
    positions = torch.split(pos, 3, dim=-1)
    edge_sh = []
    for block in positions: 
        edge_vec = block[src] - block[dst]
        edge_sh.append(self.sh(edge_vec))
    edge_sh = torch.cat(edge_sh, dim=-1)  # Concatenate spherical harmonics
```

#### 2. E3ConvConditionalSpatioTemporal

**Purpose**: Specialized for spatiotemporal conditioning where input combines physical positions with spatial features

**Key Design**:
- Expects input: `[y.pos, spatial_features]` concatenated along feature dimension
- `N_structures = 1` (processes combined input as single structure)
- `input_attr_irreps`: Defines irreps for spatial features component

**Forward Pass Logic**:
```python
def forward(self, pos: Tensor, topology, c_noise, effective_radial_cutoff):
    # Split positions: first 3 coords are physical, rest are spatial features
    pos_physical = pos[:, :3]    # [N, 3] - physical coordinates
    pos_features = pos[:, 3:]    # [N, spatial_features_dim] - spatial features
    
    # Compute edge spherical harmonics ONLY for physical positions
    edge_vec_physical = pos_physical[src] - pos_physical[dst]
    edge_sh = self.sh(edge_vec_physical)
    
    # Combine node_attr with spatial features as input attributes
    # ...
```

### Configuration Examples

**Standard Conditional Model**:
```yaml
arch:
  _target_: jamun.model.arch.E3ConvConditional
  N_structures: 2  # [current_pos, conditioned_structure]
  irreps_sh: "1x0e + 1x1e"
```

**Spatiotemporal Conditional Model**:
```yaml  
arch:
  _target_: jamun.model.arch.e3conv_conditional.E3ConvConditionalSpatioTemporal
  N_structures: 1  # Combined [y.pos, spatial_features]
  input_attr_irreps: "120x0e + 32x1e"  # Match spatiotemporal output
```

---

## Memory-Based Sampling: BAOAB/ABOBA Subroutines

### Overview

The conditional-gen branch introduces memory-enhanced versions of BAOAB and ABOBA splitting schemes that maintain and update a history of molecular states during sampling.

### Core Memory Concepts

#### 1. Historical State Management
- `y_hist`: List of previous molecular configurations
- States are maintained in chronological order: `[newest, ..., oldest]`
- History updates occur at configurable intervals via `history_update_frequency`

#### 2. Conditional Density Equilibration
The key innovation is equilibration to conditional densities `p(y_t | y_hist)` rather than marginal densities.

### BAOAB_Memory Algorithm

#### Algorithm Structure
```python
def baoab_memory(y, y_hist, score_fn, steps, history_update_frequency=1, ...):
    """BAOAB splitting scheme that updates a state history."""
```

#### Key Parameters
- `y`: Current molecular state
- `y_hist`: History of previous states  
- `score_fn`: Score function that accepts both `y` and `y_hist`
- `history_update_frequency`: How often to update the history (inner loop iterations)

#### Main Algorithm Loop

**Outer Loop** (Main sampling steps):
```python
for i in steps_iter:
    # Inner equilibration loop
    for j in range(1, history_update_frequency):
        # BAOAB step for conditional density p(y_t | y_hist)
        y_current = y.clone().detach()
        v = v + u * (delta / 2) * psi     # B step
        y = y + (delta / 2) * v           # A step  
        R = torch.randn_like(y)
        vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R  # O step
        y = y + (delta / 2) * vhat        # A step
        psi, orig_score = score_fn_processed(y, y_hist=y_hist)
        v = vhat + (delta / 2) * psi      # B step
    
    # Update history
    y_hist.pop(-1)              # Remove oldest state
    y_hist.insert(0, y_current) # Add current state as newest
```

### Inner Loop Mechanics: The Role of Equilibration

#### Purpose of the Inner Loop

Rather than taking a single MCMC step, the inner loop allows the system to equilibrate to the conditional distribution `p(y_t | y_hist)` given the fixed history.


#### Mathematical Justification
```
Standard MCMC: y_{t+1} ~ p(y | y_t)
Memory MCMC:   y_{t+1} ~ p(y | y_hist) after equilibration to p(y | y_hist)
```

### ABOBA_Memory Algorithm

Similar structure to BAOAB_memory but with ABOBA splitting:

```python
def aboba_memory(y, y_hist, score_fn, steps, history_update_frequency=1, ...):
    """ABOBA splitting scheme that updates a state history."""
    for i in steps_iter:
        for j in range(1, history_update_frequency):
            # ABOBA inner loop for equilibration
            y_current = y.clone().detach()
            y = y + (delta / 2) * v                    # A step
            psi, orig_score = score_fn_processed(y, y_hist=y_hist)
            v = v + u * (delta / 2) * psi              # B step
            R = torch.randn_like(y)
            vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R  # O step
            v = vhat + (delta / 2) * psi               # B step  
            y = y + (delta / 2) * v                    # A step
```

### Cleanup Option

Both algorithms support a `cleanup` option for denoising:

```python
if cleanup is not None and cleanup and sigma is not None:
    y_current = y.clone().detach()
    _, orig_score = score_fn_processed(y_current, y_hist=y_hist)
    y_denoised_and_noised = y_current + (sigma**2)*orig_score + sigma*torch.randn_like(y_current)
    y_hist.pop(-1)
    y_hist.insert(0, y_denoised_and_noised)
    y = y_denoised_and_noised
```

This performs a denoising step followed by re-noising before adding to history.

### Configuration

```yaml
batch_sampler:
  _target_: jamun.sampling.mcmc.BAOAB_memory
  delta: ${delta}
  friction: ${friction} 
  steps: ${num_sampling_steps_per_batch}
  history_update_frequency: 10  # Inner loop iterations
  save_trajectory: true
  cpu_offload: true
  verbose: true
```

---

## Sampling Memory Wrapper

### ModelSamplingWrapperMemory

The `ModelSamplingWrapperMemory` class extends the standard sampling wrapper to handle models that depend on historical states.

#### Key Features

#### 1. History-Aware Initialization
```python
def __init__(self, model, init_graphs, sigma, recenter_on_init=True):
    # Apply mean centering to positions
    self.init_graphs = mean_center(self.init_graphs)
    
    # Mean center hidden states if they exist
    if hasattr(self.init_graphs, 'hidden_state') and self.init_graphs.hidden_state:
        for i in range(len(self.init_graphs.hidden_state)):
            mean = scatter(self.init_graphs.hidden_state[i], self.init_graphs.batch, dim=0, reduce="mean")
            self.init_graphs.hidden_state[i] = self.init_graphs.hidden_state[i] - mean[self.init_graphs.batch]
```

#### 2. History Sampling
```python
def sample_initial_noisy_history(self) -> list:
    """Sample initial noisy history from hidden states."""
    noisy_history = []
    for hidden_state in self.init_graphs.hidden_state:
        noisy_history.append(hidden_state + torch.randn_like(hidden_state) * self.sigma)
    return noisy_history
```

#### 3. Memory-Aware Score and Prediction Functions
```python
def score(self, y, y_hist, sigma):
    """Score function that includes history."""
    graph = self.positions_to_graph(y, y_hist).to(self.device)
    return self._model.score(graph, sigma)

def xhat(self, y, y_hist, sigma):
    """Prediction function that includes history."""
    graph = self.positions_to_graph(y, y_hist).to(self.device)
    xhat_graph = self._model.xhat(graph, sigma)
    return xhat_graph.pos
```

#### 4. Graph Construction with History
```python
def positions_to_graph(self, positions: torch.Tensor, y_hist: list) -> torch_geometric.data.Data:
    """Wraps positions to a graph and attaches the historical states."""
    input_graph = self.init_graphs.clone()
    input_graph.pos = positions
    input_graph.hidden_state = y_hist  # Attach history as hidden_state
    return input_graph.to(positions.device)
```

#### 5. History-Aware Sample Unbatching
The wrapper handles complex unbatching of trajectory data that includes historical information:

```python
def unbatch_samples(self, samples: dict[str, torch.Tensor]):
    """Unbatch samples including history trajectories."""
    for key, value in samples.items():
        if key == "y_hist" or key == "y_hist_traj":
            # Special handling for history data
            if key == "y_hist":
                value = [value]
            value = torch.stack([torch.stack(traj, dim=1) for traj in value], dim=1)
            # ... complex unbatching logic for history
```

### Integration with SamplerMemory

The `SamplerMemory` class uses `ModelSamplingWrapperMemory`:

```python
class SamplerMemory(Sampler):
    def sample(self, model, batch_sampler, num_batches, init_graphs, continue_chain=False):
        model_wrapped = utils.ModelSamplingWrapperMemory(
            model=model,
            init_graphs=init_graphs,
            sigma=batch_sampler.sigma,
        )
        
        y_init = model_wrapped.sample_initial_noisy_positions()
        y_hist_init = model_wrapped.sample_initial_noisy_history()
        
        # Memory-aware sampling
        out = batch_sampler.sample(model=model_wrapped, y_init=y_init, 
                                 v_init=v_init, y_hist_init=y_hist_init)
```

### Chain Continuation

The memory wrapper supports continuing chains across batches:

```python
if continue_chain:
    y_init = out["y"]
    v_init = out["v"] 
    y_hist_init = out["y_hist"]  # Continue with updated history
else:
    y_init = model_wrapped.sample_initial_noisy_positions()
    y_hist_init = model_wrapped.sample_initial_noisy_history()
    v_init = "gaussian"
```

---

## Usage Examples

### Training a Spatiotemporal Conditional Model

```yaml
model:
  _target_: jamun.model.denoiser_conditional.Denoiser
  arch:
    _target_: jamun.model.arch.e3conv_conditional.E3ConvConditionalSpatioTemporal
    N_structures: 1
    input_attr_irreps: "120x0e + 32x1e"
  conditioner:
    _target_: jamun.model.conditioners.SpatioTemporalConditioner
    spatiotemporal_model:
      _target_: jamun.model.arch.spatiotemporal.E3SpatioTemporal
      spatial_module: # ... spatial E3Conv config
      temporal_module: # ... temporal E3Transformer config
```

### Memory-Based Sampling Configuration

```yaml
sampler:
  _target_: jamun.sampling.SamplerMemory
  devices: 1

batch_sampler:
  _target_: jamun.sampling.mcmc.BAOAB_memory
  delta: 0.04
  friction: 1.0
  steps: 1000
  history_update_frequency: 10
  save_trajectory: true
```

### Data Loading for Memory Models

```yaml
init_datasets:
  _target_: jamun.data.parse_repeated_position_datasets_from_directory
  root: "/path/to/data"
  total_lag_time: 5        # Number of historical states
  lag_subsample_rate: 1    # Temporal subsampling
  subsample: 1            # Spatial subsampling
```

This comprehensive documentation covers the key innovations in the conditional-gen branch, providing both conceptual understanding and practical implementation details for spatiotemporal modeling and memory-based sampling in molecular dynamics.

