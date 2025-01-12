import logging
import os
import re
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np
import pdbfixer
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NoseHooverIntegrator,
    PeriodicTorsionForce,
    Vec3,
)
from openmm.app import (
    PME,
    CheckpointReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
    Topology,
)
from openmm.unit import (
    angstroms,
    bar,
    kelvin,
    kilocalories_per_mole,
    kilojoules_per_mole,
    nanometer,
    nanometers,
    picoseconds,
)

Positions = List[Tuple[Vec3, ...]]
Velocities = List[Tuple[Vec3, ...]]

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("openmm_utils")


def filename_with_prefix(prefix: str, extension: str) -> str:
    """
    Generates a filename with a prefix and an integer identifier.
    Filename format: <string_indentifier>_<int_identifier>.<extension>
    """
    try:
        dir, prefix = prefix.rsplit("/", 1)
        os.makedirs(dir, exist_ok=True)
    except ValueError:
        dir = "."

    # Find last file with the same prefix and increment the integer identifier.
    files = [f for f in os.listdir(dir) if f.startswith(prefix) and f.endswith(extension)]
    max_id = -1
    for filename in files:
        search = re.search(rf"{prefix}_(\d+)", filename)
        if search is None:
            continue
        file_id = int(search.group(1))
        max_id = max(max_id, file_id)

    next_id = max_id + 1
    filename = f"{prefix}_{next_id}.{extension}"
    return os.path.join(dir, filename)


def fix_pdb(
    pdb_file: str,
    output_file_prefix: str = "fixed",
    save_file: bool = True,
) -> Tuple[Positions, Topology]:
    """Fixes the raw .pdb from colabfold using pdbfixer."""
    py_logger.info("Fixing the PDB file with pdbfixer.")
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)

    if save_file:
        output_file = filename_with_prefix(output_file_prefix, extension="pdb")
        with open(output_file, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return fixer.positions, fixer.topology


def add_hydrogens(
    positions: Positions,
    topology: Topology,
    forcefield: ForceField,
    save_file: bool = True,
    output_file_prefix: str = "hydrogenated",
) -> Tuple[Positions, Topology]:
    """Adds missing hydrogen to the pdb for a particular forcefield."""
    py_logger.info("Adding hydrogens to the system.")
    modeller = Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    if save_file:
        output_file = filename_with_prefix(output_file_prefix, extension="pdb")
        with open(output_file, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
        py_logger.info(f"Hydrogenated PDB file saved at: {os.path.abspath(output_file)}")

    return modeller.positions, modeller.topology


def solvate(
    positions: Positions,
    topology: Topology,
    forcefield: ForceField,
    padding_nm: int,
    water_model: str,
    positive_ion: str,
    negative_ion: str,
    save_file: bool = True,
    output_file_prefix: str = "solvated",
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

    if save_file:
        output_file = filename_with_prefix(output_file_prefix, extension="pdb")
        with open(output_file, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
        py_logger.info(f"Solvated PDB file saved at: {os.path.abspath(output_file)}")

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
    simulation = Simulation(topology, system, integrator)
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


def add_dihedral_restraints(dihedrals: np.ndarray, simulation: Simulation, k: float = 1) -> Simulation:
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
    positions: Positions,
    simulation: Simulation,
    tolerance_kJ_per_mol_per_nm: float = 10,
    num_steps: int = 1500,
    output_file_prefix: str = "minimized",
    save_file: bool = True,
) -> Tuple[Positions, Simulation]:
    """Energy minimization steps to relax the system."""
    py_logger.info("Minimizing the energy of the system.")
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(
        tolerance=tolerance_kJ_per_mol_per_nm * kilojoules_per_mole / nanometer, maxIterations=num_steps
    )
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()

    if save_file:
        pdb_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

        output_file = filename_with_prefix(output_file_prefix, extension="pdb")
        with open(output_file, "w") as f:
            PDBFile.writeFile(simulation.topology, pdb_positions, f)

        py_logger.info(f"Minimized PDB file saved at: {os.path.abspath(output_file)}")

    return minimized_positions, simulation


def run_simulation(
    simulation: Simulation,
    positions: Positions,
    num_steps: int,
    output_file_prefix: str,
    ensemble: str,
    output_frequency: int,
    temp_K: Optional[float] = None,
    pressure_bar: Optional[float] = None,
    save_intermediate_files: bool = False,
    velocities: Optional[Velocities] = None,
    restart_from_checkpoint: bool = False,
    save_xtc: bool = False,
    xtc_output_file: Optional[str] = None,
    save_pdb: bool = False,
    pdb_output_file: Optional[str] = None,
    save_checkpoint_file: bool = False,
    checkpoint_file: Optional[str] = None,
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
        output_frequency: Frequency of output writing (in steps)
        save_xtc: Whether to write XTC trajectory
        velocities: Optional initial velocities
        restart_from_checkpoint: Whether to restart from a checkpoint
        checkpoint_file: Name of checkpoint file to read/write

    Returns:
        Tuple of (final positions, final velocities, simulation)
    """
    # Setup ensemble forces.
    if ensemble == "NPT":
        py_logger.info(
            f"Setting up NPT ensemble with pressure barostat at {pressure_bar} bar and temperature {temp_K} K."
        )
        simulation.context.getSystem().addForce(MonteCarloBarostat(pressure_bar * bar, temp_K * kelvin))
    elif ensemble == "NVT":
        py_logger.info("Setting up NVT ensemble.")
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
        if checkpoint_file is None:
            checkpoint_file = filename_with_prefix(output_file_prefix, extension="chk")

        chkpt_freq = int(0.05 * num_steps)
        simulation.reporters.append(CheckpointReporter(checkpoint_file, chkpt_freq))

    # State reporter for logging.
    if save_intermediate_files:
        log_file = filename_with_prefix(output_file_prefix, extension="log")
        mode = "a" if restart_from_checkpoint else "w"
        simulation.reporters.append(
            StateDataReporter(
                open(log_file, mode),
                output_frequency,
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
    if save_xtc:
        if xtc_output_file is None:
            xtc_output_file = filename_with_prefix(output_file_prefix, extension="xtc")

        topology = md.Topology.from_openmm(simulation.topology)
        protein_indices = topology.select("protein")
        simulation.reporters.append(
            md.reporters.XTCReporter(xtc_output_file, output_frequency, atomSubset=protein_indices)
        )

    # Run simulation.
    simulation.step(num_steps)
    py_logger.info("Simulation completed.")

    # Save final state.
    pdb_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    if save_pdb:
        if pdb_output_file is None:
            pdb_output_file = filename_with_prefix(output_file_prefix, extension="pdb")
        with open(pdb_output_file, "w") as f:
            PDBFile.writeFile(simulation.topology, pdb_positions, f)
        py_logger.info(f"PDB file saved at: {os.path.abspath(pdb_output_file)}")

    if save_intermediate_files:
        output_file = filename_with_prefix(output_file_prefix, extension="state")
        simulation.saveState(output_file)

    if save_xtc:
        py_logger.info(f"Trajectory file saved at: {os.path.abspath(xtc_output_file)}")

    # Get final positions and velocities.
    final_positions = simulation.context.getState(getPositions=True).getPositions()

    final_velocities = simulation.context.getState(getVelocities=True).getVelocities()

    # Cleanup NPT forces.
    if ensemble == "NPT":
        simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces() - 1)

    return final_positions, final_velocities, simulation
