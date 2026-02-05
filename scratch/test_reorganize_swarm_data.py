#!/usr/bin/env python3
"""
Test script for the swarm data reorganization.

This script tests the reorganization functionality on a small subset of data
before running the full reorganization.
"""

import os
import sys
import tempfile

# Add the scratch directory to the path so we can import the main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mdtraj as md

    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("⚠️  mdtraj not available. Trajectory validation will be skipped.")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Progress bars will be disabled.")


def create_test_data(test_source_dir: str, test_pdb_file: str):
    """Create a small test dataset structure."""
    print("Creating test data structure...")

    # Create test source directory
    os.makedirs(test_source_dir, exist_ok=True)

    # Create a few test grid directories with mock files
    test_grid_codes = ["000", "001", "002", "003", "004"]
    trajectory_codes = ["001", "002", "003", "004", "005"]

    for grid_code in test_grid_codes:
        grid_dir = os.path.join(test_source_dir, f"AA_{grid_code}")
        os.makedirs(grid_dir, exist_ok=True)

        # Create mock .xtc files
        for traj_code in trajectory_codes:
            xtc_file = os.path.join(grid_dir, f"swarm_1ps_{traj_code}.xtc")
            # Create empty files that will be copied, but note they won't be valid XTC format
            # This is just for testing the file organization logic
            with open(xtc_file, "w") as f:
                f.write(f"Mock XTC data for grid {grid_code}, trajectory {traj_code}\n")

    # Create mock single PDB file with more realistic content
    os.makedirs(os.path.dirname(test_pdb_file), exist_ok=True)
    with open(test_pdb_file, "w") as f:
        # Write a minimal but valid PDB structure for 2 alanine residues
        f.write("TITLE     MOCK ALA-ALA DIPEPTIDE\n")
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n")
        f.write("ATOM      4  O   ALA A   1       1.332   2.445   0.000  1.00  0.00           O\n")
        f.write("ATOM      5  CB  ALA A   1       1.978  -0.750   1.202  1.00  0.00           C\n")
        f.write("ATOM      6  H   ALA A   1      -0.481   0.000   0.890  1.00  0.00           H\n")
        f.write("ATOM      7  HA  ALA A   1       1.804  -0.531  -0.900  1.00  0.00           H\n")
        f.write("ATOM      8  HB1 ALA A   1       1.642  -1.785   1.202  1.00  0.00           H\n")
        f.write("ATOM      9  HB2 ALA A   1       3.068  -0.750   1.202  1.00  0.00           H\n")
        f.write("ATOM     10  HB3 ALA A   1       1.642  -0.281   2.132  1.00  0.00           H\n")
        f.write("ATOM     11  N   ALA A   2       3.332   1.420   0.000  1.00  0.00           N\n")
        f.write("ATOM     12  CA  ALA A   2       4.009   2.709   0.000  1.00  0.00           C\n")
        f.write("ATOM     13  C   ALA A   2       5.509   2.709   0.000  1.00  0.00           C\n")
        f.write("ATOM     14  O   ALA A   2       6.134   1.649   0.000  1.00  0.00           O\n")
        f.write("ATOM     15  CB  ALA A   2       3.489   3.459   1.202  1.00  0.00           C\n")
        f.write("ATOM     16  H   ALA A   2       3.855   0.556   0.000  1.00  0.00           H\n")
        f.write("ATOM     17  HA  ALA A   2       3.673   3.240  -0.900  1.00  0.00           H\n")
        f.write("ATOM     18  HB1 ALA A   2       3.825   4.494   1.202  1.00  0.00           H\n")
        f.write("ATOM     19  HB2 ALA A   2       2.399   3.459   1.202  1.00  0.00           H\n")
        f.write("ATOM     20  HB3 ALA A   2       3.825   2.990   2.132  1.00  0.00           H\n")
        f.write("ATOM     21  OXT ALA A   2       6.032   3.829   0.000  1.00  0.00           O\n")
        f.write("TER      22      ALA A   2\n")
        f.write("END\n")

    print(f"Created test data with {len(test_grid_codes)} grid codes")


def test_mdtraj_with_mock_data(train_dir: str, val_dir: str):
    """
    Test mdtraj functionality with mock data.
    Note: This will likely fail since we're creating mock XTC files that aren't real trajectories.
    """
    if not MDTRAJ_AVAILABLE:
        print("⚠️  mdtraj not available, skipping trajectory compatibility tests")
        return

    print("Testing mdtraj compatibility (expected to fail with mock data)...")

    for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
        xtc_files = [f for f in os.listdir(split_dir) if f.endswith(".xtc")]
        if not xtc_files:
            continue

        # Test one file from each split
        test_file = xtc_files[0]
        base_name = test_file.replace(".xtc", "")
        pdb_file = f"{base_name}.pdb"

        xtc_path = os.path.join(split_dir, test_file)
        pdb_path = os.path.join(split_dir, pdb_file)

        try:
            # This will likely fail since we have mock XTC data
            traj = md.load(xtc_path, top=pdb_path)
            print(
                f"✅ {split_name}: Successfully loaded {test_file} + {pdb_file} "
                f"({traj.n_frames} frames, {traj.n_atoms} atoms)"
            )
            del traj
        except Exception as e:
            print(f"❌ {split_name}: Failed to load {test_file} + {pdb_file}: {str(e)}")
            print("  (This is expected with mock data - real data should work)")

    print("Note: mdtraj tests with mock data are expected to fail.")
    print("The real script will test with actual trajectory files.")


def test_reorganization():
    """Test the reorganization script with mock data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up test paths
        test_source = os.path.join(temp_dir, "test_swarm_results")
        test_target = os.path.join(temp_dir, "test_enhanced")
        test_pdb = os.path.join(temp_dir, "test_ALA_ALA.pdb")

        # Create test data
        create_test_data(test_source, test_pdb)

        # Import and modify the main script for testing
        import reorganize_swarm_data

        # Temporarily override the configuration
        original_source = reorganize_swarm_data.SOURCE_DIR
        original_pdb = reorganize_swarm_data.SINGLE_PDB_FILE
        original_strategies = reorganize_swarm_data.SPLITTING_STRATEGIES.copy()

        reorganize_swarm_data.SOURCE_DIR = test_source
        reorganize_swarm_data.SINGLE_PDB_FILE = test_pdb

        # Override the splitting strategies for testing
        reorganize_swarm_data.SPLITTING_STRATEGIES = {
            "grid_split": {
                "target_dir": test_target + "_grid_split",
                "train_size": 3,  # Use 3 for train, 2 for val
                "description": "Test grid split",
            },
            "trajectory_split": {
                "target_dir": test_target + "_trajectory_split",
                "train_trajectories": ["001", "002", "003"],  # First 3 for testing
                "val_trajectories": ["004", "005"],  # Last 2 for testing
                "description": "Test trajectory split",
            },
        }

        try:
            print("\n" + "=" * 50)
            print("RUNNING TEST REORGANIZATION")
            print("=" * 50)

            # Test both strategies
            print("Testing grid split strategy...")
            reorganize_swarm_data.main("grid_split")

            print("Testing trajectory split strategy...")
            reorganize_swarm_data.main("trajectory_split")

            # Verify results
            print("\n" + "=" * 50)
            print("VERIFYING TEST RESULTS")
            print("=" * 50)

            # Check both strategies
            for strategy_name in ["grid_split", "trajectory_split"]:
                strategy_dir = test_target + f"_{strategy_name}"
                train_dir = os.path.join(strategy_dir, "train")
                val_dir = os.path.join(strategy_dir, "val")

                if os.path.exists(train_dir) and os.path.exists(val_dir):
                    train_files = os.listdir(train_dir)
                    val_files = os.listdir(val_dir)

                    train_xtc = [f for f in train_files if f.endswith(".xtc")]
                    train_pdb = [f for f in train_files if f.endswith(".pdb")]
                    val_xtc = [f for f in val_files if f.endswith(".xtc")]
                    val_pdb = [f for f in val_files if f.endswith(".pdb")]

                    print(f"\n{strategy_name.upper()} STRATEGY:")
                    print(f"Train directory: {len(train_files)} files ({len(train_xtc)} .xtc, {len(train_pdb)} .pdb)")
                    print(f"Val directory: {len(val_files)} files ({len(val_xtc)} .xtc, {len(val_pdb)} .pdb)")

                    # Calculate expected files based on strategy
                    if strategy_name == "grid_split":
                        # 3 grid codes × 5 trajectories × 2 file types = 30 train files
                        # 2 grid codes × 5 trajectories × 2 file types = 20 val files
                        expected_train = 3 * 5 * 2
                        expected_val = 2 * 5 * 2
                    else:  # trajectory_split
                        # 5 grid codes × 3 trajectories × 2 file types = 30 train files
                        # 5 grid codes × 2 trajectories × 2 file types = 20 val files
                        expected_train = 5 * 3 * 2
                        expected_val = 5 * 2 * 2

                    if len(train_files) == expected_train and len(val_files) == expected_val:
                        print(f"✅ {strategy_name} Test PASSED! File counts are correct.")

                        # Check a few file names
                        print("Sample train files:", sorted(train_files)[:3])
                        print("Sample val files:", sorted(val_files)[:3])
                    else:
                        print(
                            f"❌ {strategy_name} Test FAILED! Expected {expected_train} train, {expected_val} val files"
                        )
                        return False
                else:
                    print(f"❌ {strategy_name} Test FAILED! Output directories were not created")
                    return False

            # Test mdtraj compatibility on one strategy
            strategy_dir = test_target + "_grid_split"
            train_dir = os.path.join(strategy_dir, "train")
            val_dir = os.path.join(strategy_dir, "val")
            print("\n=== Testing mdtraj compatibility ===")
            test_mdtraj_with_mock_data(train_dir, val_dir)

            print("\n✅ All tests completed successfully!")
            return True

        finally:
            # Restore original configuration
            reorganize_swarm_data.SOURCE_DIR = original_source
            reorganize_swarm_data.SINGLE_PDB_FILE = original_pdb
            reorganize_swarm_data.SPLITTING_STRATEGIES = original_strategies


def main():
    """Run the test."""
    print("Starting test of swarm data reorganization script...")

    success = test_reorganization()

    if success:
        print("\n🎉 Test passed! The script should work correctly on the real data.")
        print("\nTo run the full reorganization, execute:")
        print("python scratch/reorganize_swarm_data.py")
    else:
        print("\n❌ Test failed! Please check the script before running on real data.")

    return success


if __name__ == "__main__":
    main()
