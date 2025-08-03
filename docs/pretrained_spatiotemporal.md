# Pretrained Spatiotemporal Models

This document describes how to use pretrained denoiser models as spatial and temporal modules in the spatiotemporal transformer architecture.

## Overview

The spatiotemporal transformer consists of four main components:
- **Spatial Module**: Processes spatial positions (can be a pretrained denoiser)
- **Temporal Module**: Processes temporal graphs (can be a pretrained denoiser)
- **Spatial→Temporal Pooler**: Converts spatial features to temporal representation
- **Temporal→Spatial Pooler**: Converts temporal features back to spatial representation

You can now use pretrained denoiser models directly as spatial/temporal modules using the wrapper approach, with control over:
- Whether to freeze each module (trainable: true/false)
- Loading from WandB runs or direct checkpoint paths
- **No architecture matching required** - pretrained denoisers are wrapped to expose only their `xhat` function

## Quick Start

### 1. Basic Usage

The simplest way to use a pretrained denoiser is to specify it directly in your config:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /model/conditioner: spatiotemporal_pretrained_wrapper_example

model:
  conditioner:
    spatiotemporal_model:
      spatial_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "your_entity/your_project/spatial_run_id"
        trainable: false  # Freeze the pretrained spatial module
      
      temporal_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "your_entity/your_project/temporal_run_id"
        trainable: true   # Fine-tune the temporal module
```

### 2. Mixed Approach

You can also mix pretrained modules with modules trained from scratch:

```yaml
model:
  conditioner:
    spatiotemporal_model:
      # Use pretrained spatial module
      spatial_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "entity/project/spatial_run_id"
        trainable: false
      
      # Train temporal module from scratch
      temporal_module:
        _target_: jamun.model.arch.spatiotemporal.E3Transformer
        irreps_out: "3x1e"
        irreps_hidden: "8x0e + 4x1e"
        # ... other E3Transformer parameters
```

### 3. Available Config Templates

Use one of the provided config templates:

- `spatiotemporal_pretrained_wrapper_example.yaml`: Both spatial and temporal modules from pretrained denoisers
- `spatiotemporal_mixed_example.yaml`: Pretrained spatial module + temporal module trained from scratch

## Configuration Options

### Wrapper Parameters

The pretrained wrapper accepts these parameters:

```yaml
spatial_module:
  _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
  wandb_run_path: "username/project/run_id"  # Load from WandB run
  checkpoint_path: "/path/to/checkpoint.ckpt"  # OR load from direct path
  checkpoint_type: "best_so_far"  # "last", "best_so_far", or specific .ckpt filename
  trainable: true  # Whether to keep trainable (false = freeze)
```

### Key Parameters

- **`wandb_run_path`**: Load from WandB run (format: "username/project/run_id")
- **`checkpoint_path`**: Direct path to checkpoint file (mutually exclusive with wandb_run_path)
- **`checkpoint_type`**: Which checkpoint to load ("best_so_far", "last", or specific filename)
- **`trainable`**: Whether the module should be trainable (default: true, false = freeze)

## Common Use Cases

### 1. Load Pretrained Spatial, Train Temporal from Scratch

```yaml
# Use: spatiotemporal_mixed_example.yaml
model:
  conditioner:
    spatiotemporal_model:
      spatial_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "username/project/spatial_pretrained_run"
        trainable: false  # Freeze pretrained spatial
      
      temporal_module:
        _target_: jamun.model.arch.spatiotemporal.E3Transformer
        irreps_out: "3x1e"
        irreps_hidden: "8x0e + 4x1e"
        # ... standard E3Transformer configuration
```

### 2. Fine-tune Both Modules

```yaml
# Use: spatiotemporal_pretrained_wrapper_example.yaml
model:
  conditioner:
    spatiotemporal_model:
      spatial_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "username/project/spatial_run"
        trainable: true  # Fine-tune pretrained spatial
      
      temporal_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "username/project/temporal_run"
        trainable: true  # Fine-tune pretrained temporal
```

### 3. Load from Different Checkpoint Sources

```yaml
model:
  conditioner:
    spatiotemporal_model:
      spatial_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        checkpoint_path: "/shared/models/spatial_best.ckpt"
        trainable: true
      
      temporal_module:
        _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
        wandb_run_path: "username/project/temporal_run"
        trainable: true
```

## How It Works

### Wrapper Mechanism

The `DenoiserWrapper` class replicates the full denoiser logic including normalization:

```python
class DenoiserWrapper(nn.Module):
    def __init__(self, denoiser_model: nn.Module, trainable: bool = True):
        super().__init__()
        self.denoiser = denoiser_model
        
        # Control trainability
        if not trainable:
            for param in self.denoiser.parameters():
                param.requires_grad = False
                
    def forward(self, pos, topology, batch, num_graphs, c_noise, effective_radial_cutoff):
        # Replicates xhat and xhat_normalized logic from the denoiser
        # Uses denoiser's own normalization parameters
        # Returns properly normalized denoised positions
```

### Loading Process

The `return_wrapped_denoiser` function:
1. Loads the full pretrained PyTorch Lightning model from checkpoint
2. Wraps it with `DenoiserWrapper` 
3. Sets trainability based on the `trainable` parameter
4. Returns the wrapped model ready for use as a spatial/temporal module

### Advantages

- **No architecture matching required**: Works with any pretrained denoiser
- **Simple configuration**: Just specify wandb run path and trainability
- **Authentic denoiser behavior**: Replicates exact `xhat` and `xhat_normalized` logic
- **Automatic normalization**: Uses the denoiser's own training parameters
- **Flexible**: Mix pretrained and from-scratch modules easily

### Normalization Handling

The wrapper automatically uses the denoiser's own normalization parameters:

- **`normalization_type`**: The type used during training ("JAMUN", "EDM", or None)
- **`average_squared_distance`**: For JAMUN normalization 
- **`sigma_data`**: For EDM normalization
- **`mean_center`**: Whether to apply mean centering

This ensures the pretrained denoiser behaves exactly as it was trained, without any external rescaling parameters needed.

## Utility Commands (Optional)

### Inspect Checkpoint (if needed)

```bash
# Basic inspection to verify model can be loaded
python -c "
from jamun.utils.pretrained_wrapper import return_wrapped_denoiser
model = return_wrapped_denoiser(wandb_run_path='username/project/run_id')
print('✓ Model loaded successfully')
print(f'Model type: {type(model.denoiser)}')
"
```

## Module Paths

Common module paths for extracting from loaded models:

| Module | Typical Path |
|--------|-------------|
| Spatial module from spatiotemporal model | `conditioner.spatiotemporal_model.spatial_module` |
| Temporal module from spatiotemporal model | `conditioner.spatiotemporal_model.temporal_module` |
| Entire spatiotemporal model | `conditioner.spatiotemporal_model` |
| Architecture from denoiser models | `arch` |
| Conditioner from denoiser models | `conditioner` |

## Troubleshooting

### Model Loading Issues

If the checkpoint cannot be loaded as a complete model:
- Check that the checkpoint is a valid PyTorch Lightning checkpoint
- Ensure the model class can be auto-detected or specify `model_class` explicitly
- Verify that all required dependencies are available

### Module Path Errors

If a module path cannot be found:
- Use the `inspect` command to see the actual model structure
- Check that the path uses dot notation (e.g., `module.submodule`)
- Verify the checkpoint contains the expected model architecture

### Learning Rate Issues

- Use standard learning rates - freezing/unfreezing is now the main control mechanism
- Use gradient clipping when fine-tuning: `trainer.gradient_clip_val: 1.0`
- Monitor training closely in the first few epochs

### Memory Issues

- Reduce batch size when using complex spatiotemporal models
- Consider freezing larger modules if GPU memory is limited

## Example Workflows

### Workflow 1: Progressive Training

1. Train spatial module alone
2. Freeze spatial, train temporal module  
3. Fine-tune both together with low learning rates

### Workflow 2: Transfer Learning

1. Use spatial module pretrained on molecular dynamics
2. Train temporal module for your specific task
3. Fine-tune both on your target dataset

### Workflow 3: Architecture Search

1. Extract and test different pretrained modules
2. Mix and match spatial/temporal components
3. Find optimal combination for your task

## Configuration Templates

The following configuration templates are available:

- `spatiotemporal_pretrained_advanced.yaml`: Full configuration with all options
- `spatiotemporal_pretrained_spatial_only.yaml`: Load spatial only, train temporal
- `spatiotemporal_pretrained_finetune.yaml`: Fine-tune both with different learning rates
- `spatiotemporal.yaml`: Standard configuration without pretrained loading

Choose the template that best matches your use case and customize the pretrained paths.

## New Approach: Complete Model Loading

This implementation uses a **complete model loading** approach rather than state dict matching:

### Benefits:
- **No Architecture Matching**: The exact architecture from the checkpoint is loaded directly
- **Eliminates Parameter Mismatches**: No missing/unexpected keys issues
- **Preserves Model Structure**: The complete model hierarchy is maintained  
- **Automatic Class Detection**: Model classes are auto-detected from checkpoints
- **Simpler Configuration**: No need to specify complex parameter prefixes or strict loading

### How It Works:
1. **Simple Logic**: If `checkpoint_path` or `wandb_run_path` is provided → load pretrained module, otherwise use config
2. **Load Complete Model**: `Model.load_from_checkpoint()` loads the entire saved model  
3. **Extract Module**: Use `module_path` to extract the specific module (defaults to `"arch"`)
4. **Replace & Configure**: Module replaces the config version, with `trainable` flag applied

This approach is much more robust and eliminates the common issues with architecture mismatches that plague state dict loading approaches.

## Simplified Configuration

The configuration has been simplified to follow a clear principle:

**Rule: If you provide a checkpoint path → load from checkpoint, otherwise use the config architecture**

### Simple Parameters:
- **`wandb_run_path`** or **`checkpoint_path`**: Specify where to load from (if neither provided, uses config)
- **`trainable`**: `true` = trainable, `false` = freeze (default: `true`)
- **`module_path`**: What to extract from checkpoint (default: `"arch"`)

### No More:
- ❌ Complex `freeze` vs `trainable` logic
- ❌ `strict` loading parameters  
- ❌ `model_class` specifications
- ❌ Complex module prefix matching
- ❌ Architecture compatibility checking
- ❌ Learning rate multipliers and parameter groups

### Examples:
```yaml
# Load pretrained spatial, freeze it
spatial_module:
  wandb_run_path: "user/project/run_id"
  trainable: false
  
# Load pretrained temporal, fine-tune it  
temporal_module:
  checkpoint_path: "/path/to/model.ckpt"
  trainable: true
  
# No checkpoint = use config architecture
spatial_module:
  trainable: true  # Just normal training
``` 