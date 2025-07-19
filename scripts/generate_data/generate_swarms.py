#!/usr/bin/env python3

import argparse
import logging
import os
import glob
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import openmm_utils as op
from openmm.app import ForceField, Simulation, Topology

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("generate_swarms")


@dataclass
class SwarmConfig:
    """Configuration parameters for swarm trajectory generation"""
    
    # Input/Output
    input_pdbs: List[str]
    output_dir: str
    
    # MD simulation parameters (similar to run_simulation.py)
    dt_ps: float = 0.002
    temp_K: float = 300
    pressure_bar: float = 1.0
    position_restraint_k: float = 10.0  # kJ/(mol.A^2)
    forcefield: tuple[str, str] = ("amber99sbildn.xml", "tip3p.xml")
    padding_nm: float = 1.0
    water_model: str = "tip3p"
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    
    # Equilibration parameters
    energy_minimization_steps: int = 1500
    nvt_restraint_steps: int = 75_000  # Reduced from run_simulation defaults
    npt_restraint_steps: int = 75_000  # Reduced from run_simulation defaults  
    nvt_equil_steps: int = 100_000     # Reduced from run_simulation defaults
    npt_equil_steps: int = 100_000     # Reduced from run_simulation defaults
    
    # Swarm generation parameters
    num_swarms: int = 10
    swarm_steps: int = 10_000
    save_frequency: int = 10
    
    # Processing options
    save_intermediate_files: bool = False
    single_structure_mode: bool = False  # For processing just one structure
    structure_index: Optional[int] = None  # For processing a specific structure by index
    
    # New options for separated workflow
    skip_equilibration: bool = False  # Skip equilibration if already done
    equilibrate_only: bool = False    # Only do equilibration, no swarms
    append_swarms: bool = True        # Start swarm indexing from existing trajectories


def parse_args() -> SwarmConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate swarm trajectories from equilibrated structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process folder of PDBs
    %(prog)s --input-folder /path/to/pdbs --output-dir results --num-swarms 50 --swarm-steps 10000

    # Process specific PDB files
    %(prog)s --input-pdbs struct1.pdb struct2.pdb --output-dir results --num-swarms 20 --swarm-steps 5000

    # Process single structure (for parallelization)
    %(prog)s --input-pdbs struct1.pdb --output-dir results --single-structure --structure-index 1
        """,
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-folder", 
        type=str, 
        help="Folder containing PDB files to process"
    )
    input_group.add_argument(
        "--input-pdbs", 
        nargs="+", 
        help="List of PDB files to process"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for swarm trajectories"
    )

    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument(
        "--dt", type=float, default=SwarmConfig.dt_ps, 
        help="Timestep in ps (default: %(default)s)"
    )
    sim_group.add_argument(
        "--temp", type=float, default=SwarmConfig.temp_K, 
        help="Temperature in K (default: %(default)s)"
    )
    sim_group.add_argument(
        "--pressure", type=float, default=SwarmConfig.pressure_bar, 
        help="Pressure in bar (default: %(default)s)"
    )
    sim_group.add_argument(
        "--position-restraint-k", type=float, default=SwarmConfig.position_restraint_k,
        help="Position restraint force constant in kJ/(mol.A^2) (default: %(default)s)"
    )

    # Forcefield options
    ff_group = parser.add_argument_group("Forcefield Options")
    ff_group.add_argument(
        "--forcefield", nargs=2, default=SwarmConfig.forcefield,
        metavar=("FF1", "FF2"), help="Forcefield XML files (default: %(default)s)"
    )

    # Equilibration steps
    equil_group = parser.add_argument_group("Equilibration Steps")
    equil_group.add_argument(
        "--energy-minimization-steps", type=int, default=SwarmConfig.energy_minimization_steps,
        help="Steps for energy minimization (default: %(default)s)"
    )
    equil_group.add_argument(
        "--nvt-restraint-steps", type=int, default=SwarmConfig.nvt_restraint_steps,
        help="Steps for NVT equilibration with restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--npt-restraint-steps", type=int, default=SwarmConfig.npt_restraint_steps,
        help="Steps for NPT equilibration with restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--nvt-equil-steps", type=int, default=SwarmConfig.nvt_equil_steps,
        help="Steps for NVT equilibration without restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--npt-equil-steps", type=int, default=SwarmConfig.npt_equil_steps,
        help="Steps for NPT equilibration without restraints (default: %(default)s)"
    )

    # Swarm parameters
    swarm_group = parser.add_argument_group("Swarm Parameters")
    swarm_group.add_argument(
        "--num-swarms", type=int, default=SwarmConfig.num_swarms,
        help="Number of swarm trajectories to generate per structure (default: %(default)s)"
    )
    swarm_group.add_argument(
        "--swarm-steps", type=int, default=SwarmConfig.swarm_steps,
        help="Number of steps per swarm trajectory (default: %(default)s)"
    )
    swarm_group.add_argument(
        "--save-frequency", type=int, default=SwarmConfig.save_frequency,
        help="Frequency of saving frames in swarm trajectories (default: %(default)s)"
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--save-intermediate-files", action="store_true",
        help="Save intermediate files during equilibration (default: False)"
    )
    proc_group.add_argument(
        "--single-structure", action="store_true",
        help="Process only a single structure (for parallelization)"
    )
    proc_group.add_argument(
        "--structure-index", type=int,
        help="Index of structure to process (0-based, for parallelization)"
    )

    # New workflow options
    proc_group.add_argument(
        "--skip-equilibration", action="store_true",
        help="Skip equilibration if equilibrated_start.pdb already exists (default: False)"
    )
    proc_group.add_argument(
        "--equilibrate-only", action="store_true",
        help="Only perform equilibration, do not generate swarms (default: False)"
    )
    proc_group.add_argument(
        "--append-swarms", action="store_true", default=True,
        help="Start swarm indexing from existing trajectories rather than overwriting (default: True)"
    )

    args = parser.parse_args()

    # Handle input parsing
    if args.input_folder:
        # Find all PDB files in the folder
        pdb_pattern = os.path.join(args.input_folder, "*.pdb")
        input_pdbs = sorted(glob.glob(pdb_pattern))
        if not input_pdbs:
            raise ValueError(f"No PDB files found in {args.input_folder}")
        py_logger.info(f"Found {len(input_pdbs)} PDB files in {args.input_folder}")
    else:
        input_pdbs = args.input_pdbs
        # Verify all files exist
        for pdb_file in input_pdbs:
            if not os.path.exists(pdb_file):
                raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    # Handle single structure processing
    if args.single_structure:
        if args.structure_index is not None:
            if args.structure_index >= len(input_pdbs):
                raise ValueError(f"Structure index {args.structure_index} out of range (0-{len(input_pdbs)-1})")
            input_pdbs = [input_pdbs[args.structure_index]]
        else:
            if len(input_pdbs) > 1:
                py_logger.warning("Single structure mode with multiple PDbs - processing only the first one")
            input_pdbs = [input_pdbs[0]]

    return SwarmConfig(
        input_pdbs=input_pdbs,
        output_dir=args.output_dir,
        dt_ps=args.dt,
        temp_K=args.temp,
        pressure_bar=args.pressure,
        position_restraint_k=args.position_restraint_k,
        forcefield=tuple(args.forcefield),
        energy_minimization_steps=args.energy_minimization_steps,
        nvt_restraint_steps=args.nvt_restraint_steps,
        npt_restraint_steps=args.npt_restraint_steps,
        nvt_equil_steps=args.nvt_equil_steps,
        npt_equil_steps=args.npt_equil_steps,
        num_swarms=args.num_swarms,
        swarm_steps=args.swarm_steps,
        save_frequency=args.save_frequency,
        save_intermediate_files=args.save_intermediate_files,
        single_structure_mode=args.single_structure,
        structure_index=args.structure_index,
        skip_equilibration=args.skip_equilibration,
        equilibrate_only=args.equilibrate_only,
        append_swarms=args.append_swarms,
    )


def get_structure_name(pdb_file: str) -> str:
    """Get a clean structure name from PDB filename."""
    return os.path.splitext(os.path.basename(pdb_file))[0]


def setup_structure_directory(pdb_file: str, config: SwarmConfig, structure_idx: int) -> Tuple[str, str]:
    """Create output directory for a structure and return paths."""
    structure_name = get_structure_name(pdb_file)
    structure_dir = os.path.join(config.output_dir, f"AA_{structure_idx:03d}")
    
    os.makedirs(structure_dir, exist_ok=True)
    py_logger.info(f"Created structure directory: {structure_dir}")
    
    return structure_dir, structure_name


def find_existing_swarms(structure_dir: str, swarm_steps: int, dt_ps: float) -> int:
    """Find existing swarm trajectories and return the next available index."""
    trajectory_time_ps = swarm_steps * dt_ps
    pattern = os.path.join(structure_dir, f"swarm_{trajectory_time_ps:.0f}ps_*.xtc")
    existing_swarms = glob.glob(pattern)
    
    if not existing_swarms:
        return 0  # Start from 1 if no existing swarms
    
    # Extract indices from existing filenames
    indices = []
    for swarm_file in existing_swarms:
        filename = os.path.basename(swarm_file)
        # Extract index from filename like "swarm_1ps_001.xtc"
        try:
            index_part = filename.split('_')[-1].split('.')[0]  # Get "001" part
            indices.append(int(index_part))
        except (ValueError, IndexError):
            continue
    
    if indices:
        next_index = max(indices) + 1
        py_logger.info(f"Found {len(indices)} existing swarm trajectories, starting from index {next_index}")
        return next_index
    else:
        return 0


def check_equilibration_exists(structure_dir: str) -> bool:
    """Check if equilibration has already been completed."""
    equilibrated_pdb = os.path.join(structure_dir, "equilibrated_start.pdb")
    return os.path.exists(equilibrated_pdb)


def equilibrate_structure(
    pdb_file: str, 
    structure_dir: str, 
    structure_name: str, 
    config: SwarmConfig
) -> Tuple[op.Positions, op.Velocities, Simulation]:
    """
    Equilibrate a single structure starting from solvation.
    Returns the equilibrated positions, velocities, and simulation object.
    """
    py_logger.info(f"Starting equilibration for {structure_name}")
    
    # Convert to absolute path before changing directories
    pdb_file_abs = os.path.abspath(pdb_file)
    
    # Change to structure directory
    original_dir = os.getcwd()
    os.chdir(structure_dir)
    
    try:
        # Load the initial structure (assume it's already fixed and hydrogenated)
        from openmm.app import PDBFile
        pdb = PDBFile(pdb_file_abs)
        positions = pdb.positions
        topology = pdb.topology
        
        # Create forcefield
        ff = ForceField(*config.forcefield)
        
        # Solvate the structure
        py_logger.info("Solvating structure...")
        positions, topology = op.solvate(
            positions,
            topology,
            ff,
            padding_nm=config.padding_nm,
            water_model=config.water_model,
            positive_ion=config.positive_ion,
            negative_ion=config.negative_ion,
            output_file_prefix=f"{structure_name}_solvated",
            save_file=config.save_intermediate_files,
        )
        
        # Create simulation
        simulation = op.get_system_with_Langevin_integrator(
            topology, ff, config.temp_K, dt_ps=config.dt_ps
        )
        
        # Add position restraints for equilibration
        simulation = op.add_position_restraints(
            positions, topology, simulation, k=config.position_restraint_k
        )
        
        # Energy minimization
        py_logger.info("Energy minimization...")
        positions, simulation = op.minimize_energy(
            positions,
            simulation,
            num_steps=config.energy_minimization_steps,
            output_file_prefix=f"{structure_name}_minimized",
            save_file=config.save_intermediate_files,
            save_protein_only_file=False,  # Don't need protein-only file here
        )
        
        # NVT equilibration with restraints
        py_logger.info("NVT equilibration with restraints...")
        positions, velocities, simulation = op.run_simulation(
            positions=positions,
            simulation=simulation,
            velocities=None,
            output_frequency=1000,  # Less frequent output for equilibration
            save_intermediate_files=config.save_intermediate_files,
            ensemble="NVT",
            output_file_prefix=f"{structure_name}_restrainedNVT",
            num_steps=config.nvt_restraint_steps,
        )
        
        # NPT equilibration with restraints
        py_logger.info("NPT equilibration with restraints...")
        positions, velocities, simulation = op.run_simulation(
            positions=positions,
            simulation=simulation,
            velocities=velocities,
            temp_K=config.temp_K,
            pressure_bar=config.pressure_bar,
            output_frequency=1000,
            save_intermediate_files=config.save_intermediate_files,
            ensemble="NPT",
            output_file_prefix=f"{structure_name}_restrainedNPT",
            num_steps=config.npt_restraint_steps,
        )
        
        # Remove position restraints
        py_logger.info("Removing position restraints...")
        simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces() - 1)
        
        # NVT equilibration without restraints
        py_logger.info("NVT equilibration without restraints...")
        positions, velocities, simulation = op.run_simulation(
            positions=positions,
            simulation=simulation,
            velocities=velocities,
            output_frequency=1000,
            save_intermediate_files=config.save_intermediate_files,
            ensemble="NVT",
            output_file_prefix=f"{structure_name}_equilNVT",
            num_steps=config.nvt_equil_steps,
        )
        
        # Final NPT equilibration
        py_logger.info("Final NPT equilibration...")
        positions, velocities, simulation = op.run_simulation(
            positions=positions,
            simulation=simulation,
            velocities=velocities,
            temp_K=config.temp_K,
            pressure_bar=config.pressure_bar,
            output_frequency=1000,
            save_intermediate_files=config.save_intermediate_files,
            ensemble="NPT",
            output_file_prefix=f"{structure_name}_equilNPT",
            num_steps=config.npt_equil_steps,
            save_pdb=True,
            pdb_output_file="equilibrated_start.pdb",  # Save the starting structure for swarms
        )
        
        py_logger.info(f"Equilibration completed for {structure_name}")
        return positions, velocities, simulation
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)


def generate_swarms(
    positions: op.Positions,
    velocities: op.Velocities, 
    simulation: Simulation,
    structure_dir: str,
    structure_name: str,
    config: SwarmConfig
) -> None:
    """Generate swarm trajectories from equilibrated structure."""
    py_logger.info(f"Generating {config.num_swarms} swarm trajectories for {structure_name}")
    
    # Change to structure directory
    original_dir = os.getcwd()
    os.chdir(structure_dir)
    
    try:
        # Calculate trajectory time in picoseconds
        trajectory_time_ps = config.swarm_steps * config.dt_ps
        
        # Determine starting index based on existing swarms
        if config.append_swarms:
            start_idx = find_existing_swarms(structure_dir, config.swarm_steps, config.dt_ps)
        else:
            start_idx = 1
        
        for swarm_count in range(config.num_swarms):
            swarm_idx = start_idx + swarm_count
            py_logger.info(f"Generating swarm {swarm_idx + 1}/{config.num_swarms}")
            
            # Set initial conditions (same positions, slightly perturbed velocities for variation)
            simulation.context.setPositions(positions)
            
            # Add small random perturbation to velocities for each swarm
            import numpy as np
            from openmm.unit import nanometer, picosecond
            np.random.seed(swarm_idx)  # Reproducible but different per swarm
            
            # Get original velocities as numpy array
            velocities_array = np.array(velocities.value_in_unit(nanometer/picosecond))
            
            # Add small random perturbation (0.1% of thermal velocity)
            perturbation_scale = 0.001
            thermal_velocity = np.sqrt(3 * 8.314 * config.temp_K / 1000)  # Approximate thermal velocity
            perturbation = np.random.normal(0, perturbation_scale * thermal_velocity, velocities_array.shape)
            perturbed_velocities = velocities_array + perturbation
            
            # Convert back to OpenMM format
            from openmm.unit import Quantity
            perturbed_velocities_unit = Quantity(perturbed_velocities, nanometer/picosecond)
            simulation.context.setVelocities(perturbed_velocities_unit)
            
            # Generate swarm trajectory
            swarm_filename = f"swarm_{trajectory_time_ps:.0f}ps_{swarm_idx + 1:03d}.xtc"
            
            _, _, simulation = op.run_simulation(
                positions=positions,
                simulation=simulation,
                velocities=velocities,
                temp_K=config.temp_K,
                pressure_bar=config.pressure_bar,
                output_frequency=config.save_frequency,
                save_intermediate_files=False,  # No intermediate files for swarms
                ensemble="NPT",
                output_file_prefix=f"swarm_{swarm_idx + 1:03d}",
                num_steps=config.swarm_steps,
                save_xtc=True,
                xtc_output_file=swarm_filename,
                save_pdb=False,  # Don't save PDB for each swarm
            )
            
            py_logger.info(f"Completed swarm {swarm_idx + 1}: {swarm_filename}")
    
    finally:
        # Always return to original directory
        os.chdir(original_dir)


def process_structure(pdb_file: str, structure_idx: int, config: SwarmConfig) -> None:
    """Process a single structure: equilibrate and/or generate swarms."""
    structure_name = get_structure_name(pdb_file)
    py_logger.info(f"Processing structure {structure_idx + 1}: {structure_name}")
    
    # Setup structure directory
    structure_dir, structure_name = setup_structure_directory(pdb_file, config, structure_idx)
    
    # Check if equilibration exists and should be skipped
    equilibration_exists = check_equilibration_exists(structure_dir)
    
    if config.skip_equilibration and not equilibration_exists:
        py_logger.warning(f"--skip-equilibration set but no equilibrated_start.pdb found in {structure_dir}")
        py_logger.info("Proceeding with equilibration...")
        config.skip_equilibration = False
    
    # Handle equilibration
    if config.skip_equilibration and equilibration_exists:
        py_logger.info(f"Skipping equilibration for {structure_name} (equilibrated_start.pdb exists)")
        # TODO: Load from equilibrated state if needed for swarm generation
        positions, velocities, simulation = None, None, None
    else:
        # Perform equilibration
        py_logger.info(f"Starting equilibration for {structure_name}")
        positions, velocities, simulation = equilibrate_structure(
            pdb_file, structure_dir, structure_name, config
        )
    
    # Stop here if only equilibrating
    if config.equilibrate_only:
        py_logger.info(f"Equilibration-only mode: completed equilibration for {structure_name}")
        return
    
    # Generate swarms (need to implement loading from equilibrated state if skipped equilibration)
    if config.skip_equilibration and equilibration_exists:
        py_logger.info("Loading equilibrated state for swarm generation...")
        # TODO: Implement loading from saved equilibrated state
        py_logger.warning("Loading from saved equilibrated state not yet implemented!")
        py_logger.warning("Please run without --skip-equilibration for now")
        return
    
    # Generate swarm trajectories
    generate_swarms(positions, velocities, simulation, structure_dir, structure_name, config)
    
    py_logger.info(f"Completed processing structure {structure_idx + 1}: {structure_name}")


def main():
    """Main execution function."""
    config = parse_args()
    
    # Create main output directory
    os.makedirs(config.output_dir, exist_ok=True)
    py_logger.info(f"Output directory: {config.output_dir}")
    py_logger.info(f"Processing {len(config.input_pdbs)} structure(s)")
    
    # Process each structure
    for idx, pdb_file in enumerate(config.input_pdbs):
        try:
            # Use global structure index in single-structure mode, otherwise use enumeration index
            if config.single_structure_mode and config.structure_index is not None:
                structure_idx = config.structure_index
            else:
                structure_idx = idx
            
            process_structure(pdb_file, structure_idx, config)
        except Exception as e:
            py_logger.error(f"Error processing {pdb_file}: {str(e)}")
            if config.single_structure_mode:
                raise  # Re-raise in single structure mode for debugging
            else:
                py_logger.warning("Continuing with next structure...")
                continue
    
    py_logger.info("Swarm generation completed!")


if __name__ == "__main__":
    main() 