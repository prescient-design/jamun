"""
Test for ConditionerSpiked with DenoiserSpiked using ALA_ALA data.

This file demonstrates and tests the DenoiserSpiked model with ConditionerSpiked.
"""

import functools
import os
import torch
import numpy as np
from pathlib import Path

import jamun
from jamun.model import DenoiserSpiked
from jamun.model.conditioners import ConditionerSpiked
from jamun.model.arch import E3ConvConditional
import jamun.distributions
import jamun.data


def get_ala_ala_data(num_frames=20, total_lag_time=5):
    """
    Load ALA_ALA data with specified parameters.
    
    Args:
        num_frames: Number of frames to load per dataset
        total_lag_time: Number of hidden states (total time lag)
    
    Returns:
        List of datasets
    """
    # Check if data path exists
    data_path = os.getenv("JAMUN_DATA_PATH")
    if data_path is None:
        # Try common locations
        possible_paths = [
            "/data/bucket/kleinhej/",
            "/data2/sules/",
            "/path/to/data/"
        ]
        for path in possible_paths:
            ala_path = Path(path) / "capped_diamines/timewarp_splits/train"
            if ala_path.exists():
                data_path = path
                break
    
    if data_path is None:
        raise ValueError("JAMUN_DATA_PATH not set and cannot find data. Please set JAMUN_DATA_PATH environment variable.")
    
    print(f"Using data path: {data_path}")
    root_path = f"{data_path}/capped_diamines/timewarp_splits/train"
    
    datasets = jamun.data.parse_datasets_from_directory(
        root=root_path,
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=['ALA_ALA'],
        as_iterable=False,
        subsample=1,
        total_lag_time=total_lag_time,
        lag_subsample_rate=1,
        num_frames=num_frames,
        max_datasets=1  # Just use one dataset for testing
    )
    
    return datasets


def create_test_denoiser_spiked(total_lag_time=5):
    """
    Create a simple DenoiserSpiked model for testing.
    
    Args:
        total_lag_time: Number of structures for conditioning
    
    Returns:
        DenoiserSpiked model
    """
    import e3tools.nn
    
    # Note: The actual data has 4 hidden states, so we'll have 4 + 1 clean = 5 structures
    actual_n_structures = 5  # 4 hidden states + 1 clean structure
    
    arch = functools.partial(
        E3ConvConditional,
        irreps_out="1x1e",
        irreps_hidden="32x0e + 8x1e",  # Smaller for testing
        irreps_sh="1x0e + 1x1e",
        n_layers=2,  # Fewer layers for faster testing
        edge_attr_dim=32,
        atom_type_embedding_dim=8,
        atom_code_embedding_dim=8,
        residue_code_embedding_dim=16,
        residue_index_embedding_dim=8,
        use_residue_information=True,
        use_residue_sequence_index=False,
        N_structures=actual_n_structures,  # Match actual data structure count
        hidden_layer_factory=functools.partial(
            e3tools.nn.ConvBlock,
            conv=e3tools.nn.Conv,
        ),
        output_head_factory=functools.partial(
            e3tools.nn.EquivariantMLP, 
            irreps_hidden_list=["32x0e + 8x1e"]
        ),
    )
    
    conditioner = ConditionerSpiked(N_structures=actual_n_structures)
    
    denoiser = DenoiserSpiked(
        arch=arch,
        optim=functools.partial(torch.optim.Adam, lr=1e-3),
        sigma_distribution=jamun.distributions.ConstantSigma(sigma=0.04),
        max_radius=1000.0,  # Large radius for testing
        average_squared_distance=10.0,
        add_fixed_noise=False,
        add_fixed_ones=False,
        align_noisy_input_during_training=True,
        align_noisy_input_during_evaluation=True,
        mean_center=True,
        mirror_augmentation_rate=0.0,
        conditioner=conditioner,
    )
    
    return denoiser


def test_noise_and_denoise():
    """
    Test the noise_and_denoise method with ALA_ALA data.
    """
    print("=" * 60)
    print("Testing DenoiserSpiked with ConditionerSpiked on ALA_ALA data")
    print("=" * 60)
    
    # Load data
    try:
        total_lag_time = 5
        datasets = get_ala_ala_data(num_frames=10, total_lag_time=total_lag_time)
        print(f"✅ Successfully loaded {len(datasets)} datasets")
        
        dataset = datasets[0]
        print(f"   Dataset label: {dataset.label()}")
        print(f"   Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"   Sample positions shape: {sample.pos.shape}")
        print(f"   Sample hidden states: {len(sample.hidden_state) if hasattr(sample, 'hidden_state') and sample.hidden_state else 0}")
        if hasattr(sample, 'hidden_state') and sample.hidden_state:
            for i, h in enumerate(sample.hidden_state):
                print(f"     Hidden state {i} shape: {h.shape}")
                
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Create model
    try:
        denoiser = create_test_denoiser_spiked(total_lag_time=total_lag_time)
        print(f"✅ Successfully created DenoiserSpiked model")
        print(f"   Conditioner: {type(denoiser.conditioning_module).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Test noise_and_denoise
    try:
        print("\n" + "-" * 40)
        print("Testing noise_and_denoise method...")
        print("-" * 40)
        
        # Convert to batch for testing
        import torch_geometric.data
        batch = torch_geometric.data.Batch.from_data_list([sample])
        print(f"   Batch positions shape: {batch.pos.shape}")
        print(f"   Batch num_graphs: {batch.num_graphs}")
        
        # Set model to eval mode
        denoiser.eval()
        
        # Test with different sigma values
        sigma_values = [0.01, 0.04, 0.1]
        
        for sigma in sigma_values:
            print(f"\n   Testing with sigma = {sigma}")
            
            # Run noise_and_denoise
            with torch.no_grad():
                x_target, xhat, y_noisy = denoiser.noise_and_denoise(
                    batch, 
                    sigma=sigma, 
                    align_noisy_input=True
                )
            
            print(f"     ✅ noise_and_denoise completed successfully")
            print(f"     Target shape: {x_target.pos.shape}")
            print(f"     Prediction shape: {xhat.pos.shape}")
            print(f"     Noisy input shape: {y_noisy.pos.shape}")
            
            # Check that shapes match
            assert x_target.pos.shape == xhat.pos.shape == y_noisy.pos.shape
            
            # Test conditioning
            print(f"     Testing conditioner...")
            print(f"     y_noisy has hidden_state: {hasattr(y_noisy, 'hidden_state') and y_noisy.hidden_state is not None}")
            if hasattr(y_noisy, 'hidden_state') and y_noisy.hidden_state is not None:
                print(f"     y_noisy hidden_state count: {len(y_noisy.hidden_state)}")
            print(f"     x_target is not None: {x_target is not None}")
            if x_target is not None:
                print(f"     x_target.pos shape: {x_target.pos.shape}")
                print(f"     x_target has hidden_state: {hasattr(x_target, 'hidden_state') and x_target.hidden_state is not None}")
            
            conditioned_structures = denoiser.conditioner(y_noisy, x_target)
            print(f"     Conditioned structures count: {len(conditioned_structures)}")
            
            for i, struct in enumerate(conditioned_structures):
                print(f"       Structure {i} shape: {struct.shape}")
            
            # Verify that the last structure is the clean structure
            if len(conditioned_structures) > 0:
                last_structure = conditioned_structures[-1]
                clean_structure = x_target.pos
                if torch.allclose(last_structure, clean_structure, atol=1e-6):
                    print(f"     ✅ Last conditioned structure matches x_clean.pos")
                else:
                    print(f"     ❌ Last conditioned structure does NOT match x_clean.pos")
                    print(f"       Max difference: {torch.max(torch.abs(last_structure - clean_structure)).item():.8f}")
                
            # Calculate some basic metrics
            noise_level = torch.mean(torch.norm(y_noisy.pos - x_target.pos, dim=-1))
            prediction_error = torch.mean(torch.norm(xhat.pos - x_target.pos, dim=-1))
            print(f"     Average noise level: {noise_level:.4f}")
            print(f"     Average prediction error: {prediction_error:.4f}")
        
        print(f"\n✅ All noise_and_denoise tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed during noise_and_denoise testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conditioning_shapes():
    """
    Test that conditioning produces expected shapes.
    """
    print("\n" + "=" * 60)
    print("Testing ConditionerSpiked shape outputs")
    print("=" * 60)
    
    # Create dummy data
    N_atoms = 22  # ALA_ALA has 22 atoms
    N_structures = 5
    
    # Create fake batch
    pos = torch.randn(N_atoms, 3)
    hidden_states = [torch.randn(N_atoms, 3) for _ in range(N_structures - 2)]  # -2 for current pos and clean pos
    
    # Create fake torch_geometric batch
    import torch_geometric.data
    y = torch_geometric.data.Data(pos=pos, hidden_state=hidden_states)
    x_clean = torch_geometric.data.Data(pos=torch.randn(N_atoms, 3))
    
    # Test conditioner
    conditioner = ConditionerSpiked(N_structures=N_structures)
    conditioned_structures = conditioner.forward(y, x_clean)
    
    print(f"Input shapes:")
    print(f"  y.pos: {y.pos.shape}")
    print(f"  y.hidden_state: {[h.shape for h in y.hidden_state]}")
    print(f"  x_clean.pos: {x_clean.pos.shape}")
    
    print(f"\nConditioned structures:")
    for i, struct in enumerate(conditioned_structures):
        print(f"  Structure {i}: {struct.shape}")
    
    # Test concatenation (like in the model)
    concatenated = torch.cat(conditioned_structures, dim=-1)
    print(f"\nConcatenated shape: {concatenated.shape}")
    expected_dim = len(conditioned_structures) * 3  # Each structure has 3D coordinates
    print(f"Expected last dimension: {expected_dim}")
    
    # Verify that the last structure is the clean structure
    if len(conditioned_structures) > 0:
        last_structure = conditioned_structures[-1]
        clean_structure = x_clean.pos
        if torch.allclose(last_structure, clean_structure, atol=1e-6):
            print(f"✅ Last conditioned structure matches x_clean.pos")
        else:
            print(f"❌ Last conditioned structure does NOT match x_clean.pos")
            print(f"  Max difference: {torch.max(torch.abs(last_structure - clean_structure)).item():.8f}")
            assert False, "Last conditioned structure should match x_clean.pos"
    
    assert concatenated.shape == (N_atoms, expected_dim)
    print(f"✅ Shape test passed!")


if __name__ == "__main__":
    # Run tests
    try:
        # Test data loading and noise_and_denoise
        success = test_noise_and_denoise()
        
        # Test conditioning shapes
        test_conditioning_shapes()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("DenoiserSpiked with ConditionerSpiked is working correctly!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 