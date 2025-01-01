from typing import List, Tuple
import logging
import os
import re

import mdtraj as md
import numpy as np
import pdbfixer

from openmm import CustomExternalForce, MonteCarloBarostat, PeriodicTorsionForce, Platform, NoseHooverIntegrator, LangevinMiddleIntegrator
from openmm.app import PME, CheckpointReporter, HBonds, Modeller, PDBFile, Simulation, StateDataReporter, Topology, ForceField
from openmm.unit import (
    Quantity,
    angstroms,
    bar,
    kelvin,
    kilocalories_per_mole,
    kilojoules_per_mole,
    nanometer,
    nanometers,
    picoseconds,
)
Positions = Quantity[Any]

py_logger = logging.getLogger("openmm_utils")



def check_file(fname: str) -> str:
    """
    Checks the existence of a file in the given pathway and gives a newfile name
    Filename format: <string_indentifier>_<int_identifier>.<extension>
    """
    if os.path.isfile(fname):
        ident = fname.split(".")[0].split("_")
        fname = f'{ident[0]}_{int(ident[1])+1}.{fname.split(".")[1]}'
        fname = check_file(fname)
    return fname


def check_dir(fname: str) -> str:
    """
    Checks the existence of a file in the given pathway and gives a newfile name
    Filename format: <string_indentifier>_<int_identifier>.<extension>
    """
    if os.path.isdir(fname):
        ident = fname.split("_")
        ident[-1] = str(int(int(ident[-1]) + 1))
        fname = "_".join(ident)
        fname = check_dir(fname)
    return fname


def fix_pdb(pdb_file: str) -> Tuple[Positions, Topology]:
    """
    Fixes the raw .pdb from colabfold using pdbfixer.
    This needs to be performed to cleanup the pdb and to start simulation
    """
    py_logger.info("Fixing the .pdb file.")
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findNontstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    outfile = check_file(f"fixed_{pdb_file}")
    with open(outfile, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    return fixer.positions, fixer.topology


def add_hydrogens(
    positions: Positions,
    topology: Topology,
    forcefield: ForceField,
    write_file: bool = True,
) -> Tuple[Positions, Topology]:
    """Adds missing hydrogen to the pdb for a particular forcefield."""
    py_logger.info("Adding hydrogens to the system.")
    modeller = Modeller(topology, positions)
    modeller.addHydrogens(forcefield)
    if write_file:
        hydfile = check_file("fixedH_0.pdb")
        PDBFile.writeFile(modeller.topology, modeller.positions, open(hydfile, "w"))
    return modeller.positions, modeller.topology


def solvate(
    positions: Positions,
    topology: Topology,
    forcefield: ForceField,
    padding_nm: int,
    water_model: str,
    positive_ion: str,
    negative_ion: str,
    write_file: bool = True,
) -> Tuple[Positions, Topology]:
    """Creates a box of solvent with padding and neutral charges."""
    py_logger.info("Solvating the system.")
    modeller = Modeller(topology, positions)
    modeller.addSolvent(
        forcefield,
        padding=padding_nm * nanometers,
        model=water_model,
        neutralize=True,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
    )
    if write_file:
        solv_file = check_file("solvated_0.pdb")
        with open(solv_file, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
    return modeller.positions, modeller.topology


def get_system_with_Langevin_integrator(
    topology: Topology, forcefield: ForceField, temp_K: float, dt_ps: float, state: Optional[str] = None
) -> Simulation:
    """Creates a system with Langevin integrator for NVT ensemble."""
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.85 * nanometer,
        switchDistance=0.8 * nanometer,
        constraints=HBonds,
    )
    integrator = LangevinMiddleIntegrator(temp_K * kelvin, 1 / picoseconds, dt_ps * picoseconds)
    simulation = Simulation(topology, system, integrator)
    if state is not None:
        simulation.loadState(state)
    return simulation


def get_system_with_NoseHoover_integrator(
    topology: Topology, forcefield: ForceField, temp_K: float, dt_ps: float
) -> Simulation:
    """Creates a system with Nose-Hoover integrator for NVT ensemble."""
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.85 * nanometer,
        switchDistance=0.8 * nanometer,
        constraints=HBonds,
    )
    integrator = NoseHooverIntegrator(temp_K * kelvin, 1 / picoseconds, dt_ps * picoseconds)
    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(topology, system, integrator, platform)
    return simulation


def add_position_restraints(
    positions: Positions, topology: Topology, simulation: Simulation, k: float = 1000
) -> Simulation:
    """Adds an harmonic potential to the heavy atoms of the system with a force constant 'k' in units of kJ/(mol * A^2)."""

    AA = [
        "ALA",
        "ASP",
        "CYS",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ARG",
        "PRO",
        "GLN",
        "ASN",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", k * kilocalories_per_mole / angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    index = 0
    for i, res in enumerate(topology.residues()):
        # Required to select only the protein atoms
        if res.name in AA:
            for at in res.atoms():
                # All heavy atoms excluding hydrogens.
                if not re.search(r"H", at.name): 
                    force.addParticle(index, positions[index].value_in_unit(nanometers))
                index += 1

    posres_sys = simulation.context.getSystem()
    posres_sys.addForce(force)
    simulation.context.reinitialize()

    return simulation


def add_dihedral_restraints(
    dihedrals: np.ndarray, simulation: Simulation, k: float = 1
) -> Simulation:
    """Adds a periodic torsion potential to the dihedral angles of the system with a force constant 'k' in units of kJ/mol."""
    phase = [-1.1053785, -0.7255615]
    force = PeriodicTorsionForce()
    for i in [0, 1]:
        for dihed in dihedrals[i]:
            force.addTorsion(dihed[0], dihed[1], dihed[2], dihed[3], 1, phase[i], k)

    posres_sys = simulation.context.getSystem()
    posres_sys.addForce(force)
    simulation.context.reinitialize()
    return simulation


def minimize_energy(
        positions: np.ndarray,
        simulation: Simulation,
        tolerance_kJ_per_mol_per_nm: float = 10,
        n_iter: int = 1500,
        write_file: bool = True) -> Tuple[Positions, Simulation]:
    """Energy minimization steps to relax the system."""
    py_logger.info("Minimizing the energy of the system.")
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=tolerance_kJ_per_mol_per_nm * kilojoules_per_mole / nanometer, maxIterations=n_iter)
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    if write_file:
        minim_file = check_file("minim_0.pdb")
        pdb_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
        with open(minim_file, "w") as f:
            PDBFile.writeFile(simulation.topology, pdb_positions, f)
    return minimized_positions, simulation


def run_simulation(
    simulation: Simulation,
    positions: Positions,
    num_steps: int,
    name: str,
    ensemble: str,
    temp_K: Optional[float] = None,
    pressure_bar: Optional[float] = None,
    save_checkpoint_file: bool = True,
    output_frequency: int = 500,
    write_xtc: bool = True,
    velocities: Optional[Velocities] = None,
    restart_from_checkpoint: bool = False,
    checkpoint_file: str = "chkptfile.chk",
) -> Tuple[Positions, Velocities, Simulation]:
    """
    Run molecular dynamics simulation with flexible configuration options.
    
    Args:
        simulation: OpenMM Simulation object
        positions: Initial positions with units
        num_steps: Number of simulation steps
        name: Base name for output files
        ensemble: "NVT" or "NPT"
        temp_K: Temperature in Kelvin
        pressure_bar: Pressure in bar
        save_checkpoint_file: Whether to save checkpoint files
        output_frequency: Frequency of output writing
        write_xtc: Whether to write XTC trajectory
        velocities: Optional initial velocities
        restart_from_checkpoint: Whether to restart from a checkpoint
        checkpoint_file: Name of checkpoint file to read/write
    
    Returns:
        Tuple of (final positions, final velocities, simulation)
    """
    # Setup ensemble forces.
    if ensemble == "NPT":
        py_logger.info("Setting up NPT ensemble with pressure barostat.")
        simulation.context.getSystem().addForce(
            MonteCarloBarostat(pressure_bar * bar, temp_K * kelvin)
        )
    elif ensemble == "NVT":
        if pressure_bar is not None:
            py_logger.info("Ignoring pressure barostat for NVT ensemble.")
    else:
        raise ValueError(f"Invalid ensemble type: {ensemble}")

    # Set initial conditions.
    simulation.context.setPositions(positions)
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    
    if restart_from_checkpoint:
        simulation.loadCheckpoint(checkpoint_file)

    # Setup reporters.
    simulation.reporters = []
    
    # Checkpoint reporter.
    if save_checkpoint_file:
        chkpt_freq = int(0.05 * num_steps)
        simulation.reporters.append(
            CheckpointReporter(checkpoint_file, chkpt_freq)
        )

    # State reporter for logging.
    logfile = f"{name}.txt"
    with open(logfile, "a" if restart_from_checkpoint else "w") as outlog:
        simulation.reporters.append(
            StateDataReporter(
                outlog,
                output_frequency * 2,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                separator="\t|\t",
                progress=True,
                speed=True,
                totalSteps=num_steps,
            )
        )

    # XTC reporter.
    if write_xtc:
        outfname = f"{name}.xtc"
        topology = md.Topology.from_openmm(simulation.topology)
        python_expression = topology.select_expression("protein")
        req_indices = np.array(eval(python_expression))
        simulation.reporters.append(
            md.reporters.XTCReporter(outfname, output_frequency, atomSubset=req_indices)
        )

    # Run simulation.
    simulation.step(num_steps)
    py_logger.info("Simulation completed.")

    # Save final state.
    pdb_positions = simulation.context.getState(
        getPositions=True, 
        enforcePeriodicBox=True
    ).getPositions()
    
    outfile = f"{name}.pdb"
    PDBFile.writeFile(simulation.topology, pdb_positions, open(outfile, "w"))
    simulation.saveState(f"{name}.state")

    # Get final positions and velocities.
    final_positions = simulation.context.getState(
        getPositions=True
    ).getPositions()
    
    final_velocities = simulation.context.getState(
        getVelocities=True
    ).getVelocities()

    # Cleanup NPT forces.
    if ensemble == "NPT":
        simulation.context.getSystem().removeForce(
            simulation.context.getSystem().getNumForces() - 1
        )

    return final_positions, final_velocities, simulation
