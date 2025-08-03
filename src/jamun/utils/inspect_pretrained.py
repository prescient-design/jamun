#!/usr/bin/env python3
"""
Utility script for inspecting pretrained checkpoints and checking module compatibility.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import hydra
from omegaconf import DictConfig

from jamun.utils.pretrained import (
    load_checkpoint_state_dict,
    load_pretrained_model_from_checkpoint,
    extract_module_from_model,
    inspect_model_structure,
    check_model_compatibility
)
from jamun.utils.checkpoint import find_checkpoint


def print_checkpoint_structure(checkpoint_path: str, max_depth: int = 2, use_model_loading: bool = True):
    """Print the structure of a checkpoint file."""
    print(f"\n📁 Checkpoint structure: {checkpoint_path}")
    print("=" * 60)
    
    if use_model_loading:
        # Try to load as a complete model first
        try:
            model = load_pretrained_model_from_checkpoint(checkpoint_path=checkpoint_path)
            if model is not None:
                inspect_model_structure(model, max_depth)
                return
            else:
                print("⚠️  Could not load as complete model, falling back to state_dict inspection")
        except Exception as e:
            print(f"⚠️  Model loading failed ({e}), falling back to state_dict inspection")
    
    # Fallback to state_dict inspection
    try:
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        
        # Group keys by their prefixes
        key_groups: Dict[str, list] = {}
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) >= max_depth:
                prefix = '.'.join(parts[:max_depth])
            else:
                prefix = key
            
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)
        
        # Print grouped structure
        for prefix in sorted(key_groups.keys()):
            keys = key_groups[prefix]
            if len(keys) == 1 and keys[0] == prefix:
                # Single parameter
                tensor = state_dict[prefix]
                print(f"  {prefix}: {list(tensor.shape)} ({tensor.dtype})")
            else:
                # Group of parameters
                total_params = sum(state_dict[key].numel() for key in keys)
                print(f"  {prefix}.* : {len(keys)} parameters ({total_params:,} total elements)")
                
                # Show a few example keys
                if len(keys) <= 5:
                    for key in sorted(keys)[:5]:
                        tensor = state_dict[key]
                        sub_key = key[len(prefix)+1:] if key.startswith(prefix + '.') else key
                        print(f"    └─ {sub_key}: {list(tensor.shape)}")
                else:
                    for key in sorted(keys)[:3]:
                        tensor = state_dict[key]
                        sub_key = key[len(prefix)+1:] if key.startswith(prefix + '.') else key
                        print(f"    └─ {sub_key}: {list(tensor.shape)}")
                    print(f"    └─ ... and {len(keys)-3} more")
        
        total_params = sum(tensor.numel() for tensor in state_dict.values())
        print(f"\n📊 Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


def check_compatibility_with_config(
    checkpoint_path: str, 
    config_path: str, 
    module_path: Optional[str] = None
):
    """Check if a checkpoint is compatible and can be loaded."""
    print(f"\n🔍 Checking compatibility...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    if module_path:
        print(f"Module path: {module_path}")
    print("=" * 60)
    
    try:
        # Check if checkpoint can be loaded as a model
        compatibility = check_model_compatibility(checkpoint_path=checkpoint_path)
        
        if compatibility['loadable']:
            print("✅ Checkpoint can be loaded as a complete model!")
            print(f"Model class: {compatibility['model_class'].__name__}")
            print(f"Total parameters: {compatibility['total_params']:,}")
            print(f"Trainable parameters: {compatibility['trainable_params']:,}")
            
            # If a module path is specified, try to extract it
            if module_path:
                try:
                    model = load_pretrained_model_from_checkpoint(checkpoint_path=checkpoint_path)
                    extracted_module = extract_module_from_model(model, module_path)
                    if extracted_module is not None:
                        print(f"✅ Module at path '{module_path}' can be extracted!")
                        module_params = sum(p.numel() for p in extracted_module.parameters())
                        print(f"Module parameters: {module_params:,}")
                    else:
                        print(f"❌ Module at path '{module_path}' not found in model")
                except Exception as e:
                    print(f"❌ Error extracting module: {e}")
            
            # Try loading with the config if provided
            if config_path and Path(config_path).exists():
                try:
                    with hydra.initialize_config_dir(config_dir=str(Path(config_path).parent.absolute())):
                        cfg = hydra.compose(config_name=Path(config_path).stem)
                        if isinstance(cfg, DictConfig):
                            target_model = hydra.utils.instantiate(cfg)
                            if isinstance(target_model, compatibility['model_class']):
                                print("✅ Config model class matches checkpoint model class!")
                            else:
                                print(f"⚠️  Config model class ({type(target_model).__name__}) differs from checkpoint ({compatibility['model_class'].__name__})")
                except Exception as e:
                    print(f"⚠️  Could not instantiate model from config: {e}")
        else:
            print("❌ Checkpoint cannot be loaded as a model")
            if 'error' in compatibility:
                print(f"Error: {compatibility['error']}")
        
    except Exception as e:
        print(f"❌ Error checking compatibility: {e}")


def extract_and_save_module(
    checkpoint_path: str,
    module_path: str,
    output_path: str
):
    """Extract a specific module from a checkpoint and save it separately."""
    print(f"\n📤 Extracting module: {module_path}")
    print(f"From: {checkpoint_path}")
    print(f"To: {output_path}")
    print("=" * 60)
    
    try:
        # Load the full model
        model = load_pretrained_model_from_checkpoint(checkpoint_path=checkpoint_path)
        if model is None:
            print("❌ Could not load model from checkpoint")
            return
        
        # Extract the specific module
        extracted_module = extract_module_from_model(model, module_path)
        if extracted_module is None:
            print(f"❌ Module at path '{module_path}' not found in model")
            return
        
        # Save the extracted module
        # We'll save it as a state dict that can be loaded later
        module_state_dict = extracted_module.state_dict()
        
        save_data = {
            'state_dict': module_state_dict,
            'module_class': type(extracted_module).__name__,
            'module_path': module_path,
            'source_checkpoint': checkpoint_path
        }
        
        torch.save(save_data, output_path)
        
        param_count = sum(tensor.numel() for tensor in module_state_dict.values())
        print(f"✅ Extracted {len(module_state_dict)} parameters ({param_count:,} elements)")
        print(f"Module class: {type(extracted_module).__name__}")
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error extracting module: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect pretrained checkpoints for spatiotemporal models")
    parser.add_argument("command", choices=["inspect", "check", "extract"], help="Command to run")
    
    # Common arguments
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--wandb_run", type=str, help="WandB run path (e.g., user/project/run_id)")
    parser.add_argument("--checkpoint_type", type=str, default="best_so_far", 
                       help="Type of checkpoint to load")
    
    # Inspect command arguments
    parser.add_argument("--max_depth", type=int, default=2, 
                       help="Maximum depth for structure inspection")
    
    # Check command arguments
    parser.add_argument("--config", type=str, help="Path to model config file")
    parser.add_argument("--module_path", type=str, 
                       help="Module path to extract (e.g., 'conditioner.spatiotemporal_model.spatial_module')")
    
    # Extract command arguments
    parser.add_argument("--output", type=str, help="Output path for extracted module")
    
    args = parser.parse_args()
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.wandb_run:
        checkpoint_path = find_checkpoint(
            wandb_train_run_path=args.wandb_run,
            checkpoint_type=args.checkpoint_type
        )
    else:
        print("❌ Must specify either --checkpoint or --wandb_run")
        sys.exit(1)
    
    # Execute command
    if args.command == "inspect":
        print_checkpoint_structure(checkpoint_path, args.max_depth)
    
    elif args.command == "check":
        if not args.config:
            print("❌ --config is required for check command")
            sys.exit(1)
        check_compatibility_with_config(checkpoint_path, args.config, args.module_path)
    
    elif args.command == "extract":
        if not args.module_path or not args.output:
            print("❌ --module_path and --output are required for extract command")
            sys.exit(1)
        extract_and_save_module(checkpoint_path, args.module_path, args.output)


if __name__ == "__main__":
    main() 