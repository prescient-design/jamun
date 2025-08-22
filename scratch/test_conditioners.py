#!/usr/bin/env python3
"""
Test script for SelfConditioner, PositionConditioner, and MeanConditioner with both 
MDtrajDataset and RepeatedPositionDataset.
"""

import torch
import torch_geometric
import os
from pathlib import Path

# Add the src directory to the path so we can import jamun modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jamun.data._mdtraj import MDtrajDataset
from jamun.data.noisy_position_dataset import RepeatedPositionDataset
from jamun.model.conditioners.conditioners import SelfConditioner, PositionConditioner, MeanConditioner

def print_tensor_summary(tensor, name, max_elements=6):
    """Print a summary of a tensor with first few elements."""
    if tensor.numel() <= max_elements:
        print(f"{name}: {tensor.flatten().tolist()}")
    else:
        flat = tensor.flatten()
        print(f"{name} (shape {tensor.shape}): [{flat[0]:.6f}, {flat[1]:.6f}, {flat[2]:.6f}, ..., {flat[-3]:.6f}, {flat[-2]:.6f}, {flat[-1]:.6f}]")

def create_datasets():
    """Create both types of datasets with 3 total structures (2 hidden states)."""
    
    root = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train"
    traj_files = ["ALA_ALA.xtc"]
    pdb_file = "ALA_ALA.pdb"
    total_lag_time = 3  # This should create 2 hidden states (3 - 1 = 2)
    
    print(f"Creating datasets with total_lag_time={total_lag_time} (expecting 2 hidden states)...")
    
    # Create MDtrajDataset (with real lag processing)
    mdtraj_dataset = MDtrajDataset(
        root=root,
        traj_files=traj_files,
        pdb_file=pdb_file,
        label="ALA_ALA_mdtraj",
        total_lag_time=total_lag_time,
        lag_subsample_rate=1,
        num_frames=10,
        verbose=True
    )
    
    # Create RepeatedPositionDataset (with position copies)
    repeated_dataset = RepeatedPositionDataset(
        root=root,
        traj_files=traj_files,
        pdb_file=pdb_file,
        label="ALA_ALA_repeated",
        total_lag_time=total_lag_time,
        lag_subsample_rate=1,
        num_frames=10,
        verbose=True
    )
    
    return mdtraj_dataset, repeated_dataset

def create_batch_from_dataset(dataset, sample_idx=0):
    """Create a batched graph from a single dataset sample."""
    graph = dataset[sample_idx]
    batch = torch_geometric.data.Batch.from_data_list([graph])
    return batch

def print_batch_details(batch, batch_name):
    """Print detailed information about a batch."""
    print(f"\n--- {batch_name} Details ---")
    print(f"Position shape: {batch.pos.shape}")
    print_tensor_summary(batch.pos, "Position")
    
    # Check if position is mean centered
    pos_mean = torch.mean(batch.pos, dim=0)  # Mean over atoms
    pos_mean_magnitude = torch.norm(pos_mean).item()
    print(f"Position mean: [{pos_mean[0]:.6f}, {pos_mean[1]:.6f}, {pos_mean[2]:.6f}]")
    print(f"Position mean magnitude: {pos_mean_magnitude:.6f}")
    if pos_mean_magnitude < 1e-6:
        print(f"✅ Input position is mean centered")
    else:
        print(f"❌ Input position is NOT mean centered")
    
    print(f"Number of hidden states: {len(batch.hidden_state)}")
    for i, hidden_state in enumerate(batch.hidden_state):
        print(f"Hidden state {i} shape: {hidden_state.shape}")
        print_tensor_summary(hidden_state, f"Hidden state {i}")
        
        # Check if hidden state is mean centered
        hidden_mean = torch.mean(hidden_state, dim=0)  # Mean over atoms
        hidden_mean_magnitude = torch.norm(hidden_mean).item()
        print(f"Hidden state {i} mean: [{hidden_mean[0]:.6f}, {hidden_mean[1]:.6f}, {hidden_mean[2]:.6f}]")
        print(f"Hidden state {i} mean magnitude: {hidden_mean_magnitude:.6f}")
        if hidden_mean_magnitude < 1e-6:
            print(f"✅ Hidden state {i} is mean centered")
        else:
            print(f"❌ Hidden state {i} is NOT mean centered")

def test_conditioner_detailed(conditioner, batch, test_name):
    """Test a conditioner with detailed output."""
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")
    
    # Print input details
    print_batch_details(batch, "Input Batch")
    
    # Run the conditioner
    try:
        print(f"\nRunning {conditioner.__class__.__name__}...")
        conditioned_structures = conditioner(batch)
        
        print(f"\n--- Conditioner Output ---")
        print(f"Number of conditioned structures: {len(conditioned_structures)}")
        
        # Print each conditioned structure
        for i, structure in enumerate(conditioned_structures):
            print(f"\nConditioned structure {i} shape: {structure.shape}")
            print_tensor_summary(structure, f"Conditioned structure {i}")
            
            # Compare with input position
            pos_diff = torch.max(torch.abs(structure - batch.pos)).item()
            print(f"Max difference from current position: {pos_diff:.10f}")
            
            # Compare with hidden states if available
            if i < len(batch.hidden_state):
                hidden_diff = torch.max(torch.abs(structure - batch.hidden_state[i])).item()
                print(f"Max difference from hidden state {i}: {hidden_diff:.10f}")
            
            # Check if structure is mean centered (for PositionConditioner)
            if conditioner.__class__.__name__ == "PositionConditioner":
                structure_mean = torch.mean(structure, dim=0)  # Mean over atoms
                mean_magnitude = torch.norm(structure_mean).item()
                print(f"Mean of structure {i}: [{structure_mean[0]:.6f}, {structure_mean[1]:.6f}, {structure_mean[2]:.6f}]")
                print(f"Magnitude of mean: {mean_magnitude:.6f}")
                
                # Check if it's close to zero (mean centered)
                if mean_magnitude < 1e-6:
                    print(f"✅ Structure {i} is mean centered (mean ≈ 0)")
                else:
                    print(f"❌ Structure {i} is NOT mean centered")
                    
            # Check if structure contains means across time steps (for MeanConditioner)
            if conditioner.__class__.__name__ == "MeanConditioner":
                # For MeanConditioner, each structure should be the mean across time steps
                # All structures should be identical, but atoms can have different coordinates
                print(f"✅ Structure {i} contains mean across time steps")
                
                # Check if this structure is the same as the first structure (all should be the same mean)
                if i > 0:
                    first_structure = conditioned_structures[0]
                    structures_same = torch.allclose(structure, first_structure, atol=1e-6)
                    if structures_same:
                        print(f"✅ Structure {i} matches structure 0 (all structures are identical means)")
                    else:
                        print(f"❌ Structure {i} doesn't match structure 0 (all should be identical)")
                        max_diff = torch.max(torch.abs(structure - first_structure)).item()
                        print(f"Maximum difference: {max_diff:.10f}")
                
                # Verify the mean computation is correct by manually computing it
                if hasattr(batch, "hidden_state") and batch.hidden_state is not None:
                    all_positions = [batch.pos] + batch.hidden_state
                    manual_mean = torch.mean(torch.stack(all_positions, dim=0), dim=0)
                    mean_correct = torch.allclose(structure, manual_mean, atol=1e-6)
                    if mean_correct:
                        print(f"✅ Structure {i} correctly computed as mean across {len(all_positions)} time steps")
                    else:
                        print(f"❌ Structure {i} mean computation incorrect")
                        max_diff = torch.max(torch.abs(structure - manual_mean)).item()
                        print(f"Maximum difference from expected mean: {max_diff:.10f}")
                else:
                    # If no hidden states, should just be y.pos repeated
                    pos_same = torch.allclose(structure, batch.pos, atol=1e-6)
                    if pos_same:
                        print(f"✅ Structure {i} correctly equals y.pos (no hidden states)")
                    else:
                        print(f"❌ Structure {i} should equal y.pos when no hidden states")
                
        return conditioned_structures
        
    except Exception as e:
        print(f"❌ ERROR: Exception in {test_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_results(conditioned_structures, batch, test_name, expected_behavior):
    """Verify the results match expected behavior."""
    print(f"\n--- Verification for {test_name} ---")
    print(f"Expected behavior: {expected_behavior}")
    
    expected_count = 3  # N_structures = 3, so we expect 3 total structures including current position
    if len(conditioned_structures) != expected_count:
        print(f"❌ ERROR: Expected {expected_count} structures, got {len(conditioned_structures)}")
        return False
    
    print(f"✅ Correct count: {len(conditioned_structures)} structures")
    
    success = True
    for i, structure in enumerate(conditioned_structures):
        if structure.shape != batch.pos.shape:
            print(f"❌ ERROR: Structure {i} shape mismatch!")
            success = False
        else:
            print(f"✅ Structure {i} has correct shape")
    
    # Verify that the first structure is the current position for most conditioners
    # Exception: MeanConditioner returns time-averaged means, not current position
    first_structure = conditioned_structures[0]
    pos_diff = torch.max(torch.abs(first_structure - batch.pos)).item()
    
    if test_name.startswith("MeanConditioner"):
        # For MeanConditioner, first structure should be the time-averaged mean, not y.pos
        if hasattr(batch, "hidden_state") and batch.hidden_state is not None:
            all_positions = [batch.pos] + batch.hidden_state
            expected_mean = torch.mean(torch.stack(all_positions, dim=0), dim=0)
            mean_diff = torch.max(torch.abs(first_structure - expected_mean)).item()
            if mean_diff < 1e-6:
                print(f"✅ First structure correctly equals time-averaged mean (diff: {mean_diff:.2e})")
            else:
                print(f"❌ ERROR: First structure doesn't match expected time-averaged mean (diff: {mean_diff:.2e})")
                success = False
        else:
            # If no hidden states, should equal y.pos
            if pos_diff < 1e-10:
                print(f"✅ First structure correctly equals y.pos (no hidden states, diff: {pos_diff:.2e})")
            else:
                print(f"❌ ERROR: First structure doesn't match y.pos when no hidden states (diff: {pos_diff:.2e})")
                success = False
    else:
        # For other conditioners, first structure should be y.pos
        if pos_diff < 1e-10:
            print(f"✅ First structure matches current position (diff: {pos_diff:.2e})")
        else:
            print(f"❌ ERROR: First structure doesn't match current position (diff: {pos_diff:.2e})")
            success = False
    
    return success

def main():
    """Main test function."""
    print("Testing Conditioners: SelfConditioner, PositionConditioner, MeanConditioner with 3 total structures (2 hidden states)")
    print("=" * 70)
    
    # Create datasets
    try:
        mdtraj_dataset, repeated_dataset = create_datasets()
        print(f"✅ Created datasets")
        print(f"   MDtrajDataset length: {len(mdtraj_dataset)}")
        print(f"   RepeatedPositionDataset length: {len(repeated_dataset)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to create datasets: {e}")
        return False
    
    # Create batches
    try:
        mdtraj_batch = create_batch_from_dataset(mdtraj_dataset, sample_idx=0)
        repeated_batch = create_batch_from_dataset(repeated_dataset, sample_idx=0)
        print(f"✅ Created batches")
    except Exception as e:
        print(f"❌ ERROR: Failed to create batches: {e}")
        return False
    
    # Create conditioners
    try:
        self_conditioner = SelfConditioner(N_structures=3)
        position_conditioner = PositionConditioner(N_structures=3)
        mean_conditioner = MeanConditioner(N_structures=3)
        print(f"✅ Created conditioners")
    except Exception as e:
        print(f"❌ ERROR: Failed to create conditioners: {e}")
        return False
    
    # Test 1: SelfConditioner on MDtrajDataset
    result1 = test_conditioner_detailed(
        self_conditioner, 
        mdtraj_batch, 
        "TEST 1: SelfConditioner on MDtrajDataset"
    )
    if result1 is False:
        return False
    success1 = verify_results(
        result1, 
        mdtraj_batch, 
        "SelfConditioner + MDtrajDataset",
        "Should return [y.pos, y.pos, y.pos] - 3 copies of current position"
    )
    
    # Test 2: SelfConditioner on RepeatedPositionDataset  
    result2 = test_conditioner_detailed(
        self_conditioner, 
        repeated_batch, 
        "TEST 2: SelfConditioner on RepeatedPositionDataset"
    )
    if result2 is False:
        return False
    success2 = verify_results(
        result2, 
        repeated_batch, 
        "SelfConditioner + RepeatedPositionDataset",
        "Should return [y.pos, y.pos, y.pos] - 3 copies of current position"
    )
    
    # Test 3: PositionConditioner on MDtrajDataset
    result3 = test_conditioner_detailed(
        position_conditioner, 
        mdtraj_batch, 
        "TEST 3: PositionConditioner on MDtrajDataset"
    )
    if result3 is False:
        return False
    success3 = verify_results(
        result3, 
        mdtraj_batch, 
        "PositionConditioner + MDtrajDataset",
        "Should return [y.pos, aligned_hidden_state_1, aligned_hidden_state_2] - current position + 2 aligned hidden states"
    )
    
    # Test 4: PositionConditioner on RepeatedPositionDataset
    result4 = test_conditioner_detailed(
        position_conditioner, 
        repeated_batch, 
        "TEST 4: PositionConditioner on RepeatedPositionDataset"
    )
    if result4 is False:
        return False
    success4 = verify_results(
        result4, 
        repeated_batch, 
        "PositionConditioner + RepeatedPositionDataset",
        "Should return [y.pos, aligned_copy_1, aligned_copy_2] - current position + 2 aligned copies"
    )
    
    # Test 5: MeanConditioner on MDtrajDataset
    result5 = test_conditioner_detailed(
        mean_conditioner, 
        mdtraj_batch, 
        "TEST 5: MeanConditioner on MDtrajDataset"
    )
    if result5 is False:
        return False
    success5 = verify_results(
        result5, 
        mdtraj_batch, 
        "MeanConditioner + MDtrajDataset",
        "Should return [time_mean, time_mean, time_mean] - 3 copies of mean across time steps (y.pos + hidden states)"
    )
    
    # Test 6: MeanConditioner on RepeatedPositionDataset
    result6 = test_conditioner_detailed(
        mean_conditioner, 
        repeated_batch, 
        "TEST 6: MeanConditioner on RepeatedPositionDataset"
    )
    if result6 is False:
        return False
    success6 = verify_results(
        result6, 
        repeated_batch, 
        "MeanConditioner + RepeatedPositionDataset",
        "Should return [time_mean, time_mean, time_mean] - 3 copies of mean across time steps (y.pos + hidden states)"
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    tests = [
        ("SelfConditioner + MDtrajDataset", success1),
        ("SelfConditioner + RepeatedPositionDataset", success2), 
        ("PositionConditioner + MDtrajDataset", success3),
        ("PositionConditioner + RepeatedPositionDataset", success4),
        ("MeanConditioner + MDtrajDataset", success5),
        ("MeanConditioner + RepeatedPositionDataset", success6)
    ]
    
    all_passed = True
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All conditioner tests passed!")
        return True
    else:
        print(f"\n💥 Some conditioner tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 