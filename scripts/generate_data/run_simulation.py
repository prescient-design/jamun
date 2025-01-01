#!/usr/bin/env python3

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
from openmm import Positions
from openmm.app import ForceField, Simulation
import openmm_utils as op

@dataclass
class SimulationConfig:
    """Configuration parameters for MD simulation"""
    protein_dir: str
    dt_ps: float = 0.002
    temp_K: float = 300
    pressure_bar: float = 1.0
    position_position_restraint_k: float = 10.0  # kJ/(mol.A^2)
    nvt_restraint_steps: int = 150_000
    npt_restraint_steps: int = 150_000
    nvt_equil_steps: int = 250_000
    npt_equil_steps: int = 10_000_000
    forcefield: tuple[str, str] = ("amber99sbildn.xml", "tip3p.xml")


def parse_args() -> SimulationConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MD simulation with equilibration protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default parameters
    %(prog)s protein_dir

    # Custom temperature and pressure
    %(prog)s protein_dir --temp 310 --pressure 1.5

    # Using CHARMM forcefield
    %(prog)s protein_dir --forcefield charmm36.xml charmm36/water.xml

    # Full customization of equilibration steps
    %(prog)s protein_dir --nvt-restraint-steps 200000 --npt-restraint-steps 200000 \\
        --nvt-equil-steps 500000 --npt-equil-steps 5000000
        """
    )

    # Required arguments
    parser.add_argument(
        "protein_dir",
        help="Directory containing init.pdb"
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        "--dt", type=float, default=0.002,
        help="Timestep in ps (default: %(default)s)"
    )
    sim_group.add_argument(
        "--temp", type=float, default=300,
        help="Temperature in K (default: %(default)s)"
    )
    sim_group.add_argument(
        "--pressure", type=float, default=1.0,
        help="Pressure in bar (default: %(default)s)"
    )
    sim_group.add_argument(
        "--restraint-k", type=float, default=10.0,
        help="Position restraint force constant in KJ/(mol.A^2) (default: %(default)s)"
    )

    # Forcefield options
    ff_group = parser.add_argument_group('Forcefield Options')
    ff_group.add_argument(
        "--forcefield", nargs=2, 
        default=["amber99sbildn.xml", "tip3p.xml"],
        metavar=('FF1', 'FF2'),
        help="Forcefield XML files (default: %(default)s)"
    )

    # Equilibration steps
    equil_group = parser.add_argument_group('Equilibration Protocol')
    equil_group.add_argument(
        "--nvt-restraint-steps", type=int, default=150_000,
        help="Steps for NVT equilibration with restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--npt-restraint-steps", type=int, default=150_000,
        help="Steps for NPT equilibration with restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--nvt-equil-steps", type=int, default=250_000,
        help="Steps for NVT equilibration without restraints (default: %(default)s)"
    )
    equil_group.add_argument(
        "--npt-equil-steps", type=int, default=10_000_000,
        help="Steps for NPT equilibration without restraints (default: %(default)s)"
    )

    args = parser.parse_args()

    # Create and return config object
    return SimulationConfig(
        protein_dir=args.protein_dir,
        dt_ps=args.dt,
        temp_K=args.temp,
        pressure_bar=args.pressure,
        position_restraint_k=args.position_restraint_k,
        forcefield=tuple(args.forcefield),
        nvt_restraint_steps=args.nvt_restraint_steps,
        npt_restraint_steps=args.npt_restraint_steps,
        nvt_equil_steps=args.nvt_equil_steps,
        npt_equil_steps=args.npt_equil_steps
    )

def setup_system(
    config: SimulationConfig
) -> Tuple[Positions, Topology, Simulation]:
    """Set up the initial system"""
    # Fix PDB and add hydrogens
    positions, topology = op.fix_pdb("init.pdb")
    ff = ForceField(*config.forcefield)
    positions, topology = op.add_hydrogens(positions, topology, ff)
    
    # Solvate system
    positions, topology = op.solvate(
        positions, topology, ff,
        padding_nm=1,
        water_model="tip3p",
        positive_ion="Na+",
        negative_ion="Cl-",
    )
    
    # Create and setup simulation
    simulation = op.get_system_with_Langevin_integrator(
        topology, ff,
        config.temp_K,
        dt_ps=config.dt_ps
    )
    
    return positions, topology, simulation


def run_equilibration(
    positions: Positions,
    topology: Topology,
    simulation: Simulation,
    config: SimulationConfig
) -> None:
    """Run the full equilibration protocol"""
    # Add restraints.
    simulation = op.add_position_restraints(
        positions, topology,
        simulation,
        k=config.position_restraint_k
    )

    # Minimize energy.
    positions, simulation = op.minimize_energy(positions, simulation)

    # NVT with restraints.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=None,
        num_steps=config.nvt_restraint_steps,
        ensemble="NVT",
        run_type="restrainedNVT"
    )

    # NPT with restraints.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        num_steps=config.npt_restraint_steps,
        ensemble="NPT",
        run_type="restrainedNPT",
        temp_K=config.temp_K,
        pressure=config.pressure_bar
    )

    # Remove restraints.
    simulation.context.getSystem().removeForce(
        simulation.context.getSystem().getNumForces() - 1
    )

    # NVT equilibration.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        num_steps=config.nvt_equil_steps,
        ensemble="NVT",
        run_type="equilNVT"
    )

    # NPT equilibration.
    positions, velocities, simulation = op.run_simulation(
        positions=positions,
        simulation=simulation,
        velocities=velocities,
        num_steps=config.npt_equil_steps,
        ensemble="NPT",
        run_type="equilNPT",
        temp_K=config.temp_K,
        pressure=config.pressure_bar
    )


if __name__ == "__main__":
    config = parse_args()
    os.chdir(config.protein_dir)
    
    positions, simulation = setup_system(config)
    run_equilibration(positions, simulation, config)