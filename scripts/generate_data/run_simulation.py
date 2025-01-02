#!/usr/bin/env python3

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import openmm_utils as op
from openmm.app import ForceField, Simulation, Topology

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    level=logging.INFO
)
py_logger = logging.getLogger("run_simulation")

@dataclass
class SimulationConfig:
    """Configuration parameters for MD simulation"""
    init_pdb: str
    output_dir: Optional[str] = None
    dt_ps: float = 0.002
    temp_K: float = 300
    pressure_bar: float = 1.0
    position_restraint_k: float = 10.0  # kJ/(mol.A^2)
    energy_minimization_steps: int = 1500
    # nvt_restraint_steps: int = 150_000
    # npt_restraint_steps: int = 150_000
    # nvt_equil_steps: int = 250_000
    # npt_equil_steps: int = 10_000_000

    nvt_restraint_steps: int = 15_000
    npt_restraint_steps: int = 15_000
    nvt_equil_steps: int = 25_000
    npt_equil_steps: int = 20_000

    forcefield: tuple[str, str] = ("amber99sbildn.xml", "tip3p.xml")
    padding_nm: float = 1.0
    water_model: str = "tip3p"
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    output_frequency: int = 1000
    save_intermediate_files: bool = False


def parse_args() -> SimulationConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MD simulation with equilibration protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default parameters
    %(prog)s init_pdb

    # Custom temperature and pressure
    %(prog)s init_pdb --temp 310 --pressure 1.5

    # Using CHARMM forcefield
    %(prog)s init_pdb --forcefield charmm36.xml charmm36/water.xml

    # Full customization of equilibration steps
    %(prog)s init_pdb --nvt-restraint-steps 200000 --npt-restraint-steps 200000 \\
        --nvt-equil-steps 500000 --npt-equil-steps 5000000
        """
    )

    # Required arguments
    parser.add_argument(
        "init_pdb",
        help="Initial PDB file."
    )

    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for simulation files (default: temporary directory)"
    )
    parser.add_argument(
        "--save-intermediate-files", action="store_true",
        help="Save intermediate files (default: False)"
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        "--dt", type=float, default=SimulationConfig.dt_ps,
        help="Timestep in ps (default: %(default)s)"
    )
    sim_group.add_argument(
        "--temp", type=float, default=SimulationConfig.temp_K,
        help="Temperature in K (default: %(default)s)"
    )
    sim_group.add_argument(
        "--pressure", type=float, default=SimulationConfig.pressure_bar,
        help="Pressure in bar (default: %(default)s)"
    )
    sim_group.add_argument(
        "--position-restraint-k", type=float, default=SimulationConfig.position_restraint_k,
        help="Position restraint force constant in KJ/(mol.A^2) (default: %(default)s)"
    )

    # Forcefield options
    ff_group = parser.add_argument_group('Forcefield Options')
    ff_group.add_argument(
        "--forcefield", nargs=2,
        default=SimulationConfig.forcefield,
        metavar=('FF1', 'FF2'),
        help="Forcefield XML files (default: %(default)s)"
    )

    # Steps for different equilibration stages.
    steps_group = parser.add_argument_group('Simulation Steps')
    steps_group.add_argument(
        "--energy-minimization-steps", type=int, default=SimulationConfig.energy_minimization_steps,
        help="Steps for energy minimization (default: %(default)s)"
    )
    steps_group.add_argument(
        "--nvt-restraint-steps", type=int, default=SimulationConfig.nvt_restraint_steps,
        help="Steps for NVT equilibration with restraints (default: %(default)s)"
    )
    steps_group.add_argument(
        "--npt-restraint-steps", type=int, default=SimulationConfig.npt_restraint_steps,
        help="Steps for NPT equilibration with restraints (default: %(default)s)"
    )
    steps_group.add_argument(
        "--nvt-equil-steps", type=int, default=SimulationConfig.nvt_equil_steps,
        help="Steps for NVT equilibration without restraints (default: %(default)s)"
    )
    steps_group.add_argument(
        "--npt-equil-steps", type=int, default=SimulationConfig.npt_equil_steps,
        help="Steps for NPT equilibration without restraints (default: %(default)s)"
    )

    args = parser.parse_args()

    # Create and return config object
    return SimulationConfig(
        init_pdb=args.init_pdb,
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
        save_intermediate_files=args.save_intermediate_files
    )


def get_output_file_prefix(filename: str) -> str:
    """Get a prefix for output files."""
    return os.path.splitext(os.path.basename(filename))[0]


def setup_system(
    config: SimulationConfig
) -> Tuple[op.Positions, Topology, Simulation]:
    """Set up the initial system."""

    # Create a temporary directory for output files, if not provided.
    output_dir = config.output_dir
    if output_dir is None:
        py_logger.info("Creating temporary directory for simulation.")
        output_dir = tempfile.mkdtemp()

    # Change to output directory.
    py_logger.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Get a prefix for output files.
    output_file_prefix = get_output_file_prefix(config.init_pdb)

    # Fix PDB and add hydrogens
    positions, topology = op.fix_pdb(
        config.init_pdb,
        output_file_prefix=f"{output_file_prefix}_fixed",
        save_file=config.save_intermediate_files
    )
    ff = ForceField(*config.forcefield)
    positions, topology = op.add_hydrogens(
        positions, topology, ff,
        output_file_prefix=f"{output_file_prefix}_hydrogenated",
        save_file=config.save_intermediate_files
    )

    # Solvate system
    positions, topology = op.solvate(
        positions, topology, ff,
        padding_nm=config.padding_nm,
        water_model=config.water_model,
        positive_ion=config.positive_ion,
        negative_ion=config.negative_ion,
        output_file_prefix=f"{output_file_prefix}_solvated",
        save_file=config.save_intermediate_files
    )

    # Create and setup simulation
    simulation = op.get_system_with_Langevin_integrator(
        topology, ff,
        config.temp_K,
        dt_ps=config.dt_ps
    )

    return positions, topology, simulation


def run_full_simulation(
    positions: op.Positions,
    topology: Topology,
    simulation: Simulation,
    config: SimulationConfig
) -> None:
    """Run the full simulation including equilibration."""
    # Add restraints.
    simulation = op.add_position_restraints(
        positions, topology,
        simulation,
        k=config.position_restraint_k
    )

    # Get a prefix for output files.
    output_file_prefix = get_output_file_prefix(config.init_pdb)

    # Minimize energy.
    positions, simulation = op.minimize_energy(positions, simulation, num_steps=config.energy_minimization_steps, output_file_prefix=f"{output_file_prefix}_minimized", save_file=config.save_intermediate_files)

    # NVT with restraints.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=None,
        output_frequency=config.output_frequency,
        save_intermediate_files=config.save_intermediate_files,
        ensemble="NVT",
        output_file_prefix=f"{output_file_prefix}_restrainedNVT",
        num_steps=config.nvt_restraint_steps,
    )

    # NPT with restraints.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        temp_K=config.temp_K,
        pressure_bar=config.pressure_bar,
        output_frequency=config.output_frequency,
        save_intermediate_files=config.save_intermediate_files,
        ensemble="NPT",
        output_file_prefix=f"{output_file_prefix}_restrainedNPT",
        num_steps=config.npt_restraint_steps,
    )

    py_logger.info("Removing position restraints.")
    simulation.context.getSystem().removeForce(
        simulation.context.getSystem().getNumForces() - 1
    )

    # NVT equilibration.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        output_frequency=config.output_frequency,
        save_intermediate_files=config.save_intermediate_files,
        ensemble="NVT",
        output_file_prefix=f"{output_file_prefix}_equilNVT",
        num_steps=config.nvt_equil_steps,
    )

    # NPT equilibration.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        temp_K=config.temp_K,
        pressure_bar=config.pressure_bar,
        output_frequency=config.output_frequency,
        save_intermediate_files=config.save_intermediate_files,
        ensemble="NPT",
        output_file_prefix=f"{output_file_prefix}_equilNPT",
        num_steps=config.npt_equil_steps,
        save_pdb=True,
        pdb_output_file=f"{output_file_prefix}.pdb",
        save_xtc=True,
        xtc_output_file=f"{output_file_prefix}.xtc"
    )


if __name__ == "__main__":
    config = parse_args()
    positions, topology, simulation = setup_system(config)
    run_full_simulation(positions, topology, simulation, config)
