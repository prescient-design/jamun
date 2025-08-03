"""Utilities for loading pretrained models from checkpoints."""

import logging
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, Optional, Union, Any
from pathlib import Path

from jamun.utils.checkpoint import find_checkpoint

py_logger = logging.getLogger("jamun")


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and any(k.startswith('model.') for k in checkpoint.keys()):
        return checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")


def load_pretrained_model_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    wandb_run_path: Optional[str] = None,
    checkpoint_type: str = "best_so_far",
    model_class: Optional[type] = None
) -> Optional[pl.LightningModule]:
    """
    Load an entire pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Direct path to checkpoint file (mutually exclusive with wandb_run_path)
        wandb_run_path: WandB run path to find checkpoint (mutually exclusive with checkpoint_path)
        checkpoint_type: Type of checkpoint to load ("best_so_far", "last", etc.)
        model_class: Optional model class to use for loading (if checkpoint doesn't contain class info)
        
    Returns:
        Loaded model or None if loading failed
    """
    if not checkpoint_path and not wandb_run_path:
        py_logger.warning("No checkpoint path or wandb run path provided, skipping pretrained loading")
        return None
    
    if checkpoint_path and wandb_run_path:
        raise ValueError("Cannot specify both checkpoint_path and wandb_run_path")
    
    try:
        # Find the checkpoint file
        if wandb_run_path:
            checkpoint_path = find_checkpoint(
                wandb_train_run_path=wandb_run_path,
                checkpoint_type=checkpoint_type
            )
        
        py_logger.info(f"Loading pretrained model from: {checkpoint_path}")
        
        # Load the entire model from checkpoint
        if model_class:
            model = model_class.load_from_checkpoint(checkpoint_path, strict=False)
        else:
            # Try to auto-detect model class from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'hyper_parameters' in checkpoint and '_target_' in checkpoint['hyper_parameters']:
                # Try to import and use the model class from checkpoint
                import importlib
                target = checkpoint['hyper_parameters']['_target_']
                module_path, class_name = target.rsplit('.', 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                model = model_class.load_from_checkpoint(checkpoint_path, strict=False)
            else:
                py_logger.error("Cannot determine model class from checkpoint and no model_class provided")
                return None
        
        py_logger.info(f"Successfully loaded pretrained model of type {type(model).__name__}")
        return model
        
    except Exception as e:
        py_logger.error(f"Error loading pretrained model: {e}")
        return None


def extract_module_from_model(model: pl.LightningModule, module_path: str) -> Optional[nn.Module]:
    """
    Extract a specific module from a loaded model using dot notation.
    
    Args:
        model: Loaded PyTorch Lightning model
        module_path: Dot-separated path to module (e.g., "conditioner.spatiotemporal_model.spatial_module")
        
    Returns:
        Extracted module or None if not found
    """
    try:
        current = model
        for attr in module_path.split('.'):
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                py_logger.warning(f"Module path '{module_path}' not found in model")
                return None
        
        py_logger.info(f"Successfully extracted module at path: {module_path}")
        return current
        
    except Exception as e:
        py_logger.error(f"Error extracting module '{module_path}': {e}")
        return None


def load_pretrained_module_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    wandb_run_path: Optional[str] = None,
    checkpoint_type: str = "best_so_far",
    module_path: Optional[str] = None,
    model_class: Optional[type] = None
) -> Optional[nn.Module]:
    """
    Load a specific module from a pretrained model checkpoint.
    
    Args:
        checkpoint_path: Direct path to checkpoint file
        wandb_run_path: WandB run path to find checkpoint  
        checkpoint_type: Type of checkpoint to load
        module_path: Dot notation path to extract specific module (e.g., "conditioner.spatiotemporal_model.spatial_module")
        model_class: Optional model class for loading
        
    Returns:
        Extracted module or None if loading failed
    """
    # Load the full model
    model = load_pretrained_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        wandb_run_path=wandb_run_path,
        checkpoint_type=checkpoint_type,
        model_class=model_class
    )
    
    if model is None:
        return None
    
    # Extract the specific module if path provided
    if module_path:
        return extract_module_from_model(model, module_path)
    else:
        # Return the entire model if no specific module path
        return model


def inspect_model_structure(model: pl.LightningModule, max_depth: int = 3) -> None:
    """Print the structure of a loaded model."""
    print(f"\n📁 Model structure: {type(model).__name__}")
    print("=" * 60)
    
    def print_module_tree(module, prefix="", depth=0):
        if depth >= max_depth:
            return
            
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            param_count = sum(p.numel() for p in child.parameters())
            
            print(f"{'  ' * depth}├─ {name}: {type(child).__name__} ({param_count:,} params)")
            
            if depth < max_depth - 1:
                print_module_tree(child, full_name, depth + 1)
    
    print_module_tree(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")


def check_model_compatibility(
    checkpoint_path: Optional[str] = None,
    wandb_run_path: Optional[str] = None,
    checkpoint_type: str = "best_so_far",
    expected_model_class: Optional[type] = None
) -> Dict[str, Any]:
    """
    Check if a checkpoint can be loaded and optionally verify model class.
    
    Returns:
        Dict with 'loadable', 'model_class', 'error' info
    """
    try:
        if wandb_run_path:
            checkpoint_path = find_checkpoint(
                wandb_train_run_path=wandb_run_path,
                checkpoint_type=checkpoint_type
            )
        
        # Try loading the model
        model = load_pretrained_model_from_checkpoint(checkpoint_path=checkpoint_path)
        
        if model is None:
            return {
                'loadable': False,
                'model_class': None,
                'error': 'Failed to load model from checkpoint'
            }
        
        model_class = type(model)
        class_compatible = True
        
        if expected_model_class:
            class_compatible = isinstance(model, expected_model_class)
        
        return {
            'loadable': True,
            'model_class': model_class,
            'class_compatible': class_compatible,
            'checkpoint_path': checkpoint_path,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
    except Exception as e:
        return {
            'loadable': False,
            'model_class': None,
            'error': str(e)
        } 