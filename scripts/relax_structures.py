import argparse
import os

import mdtraj as md
import openmm as mm
import tqdm
from openmm import app, unit


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Minimize structures from a PDB trajectory")

    parser.add_argument("--input", "-i", required=True, help="Input trajectory file (PDB format)")

    parser.add_argument("--output", "-o", default="minimized.pdb", help="Output minimized trajectory file (PDB format)")

    parser.add_argument("--n-frames", "-n", type=int, default=None, help="Number of frames to process (default: all)")

    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Maximum number of minimization iterations (default: 10)"
    )

    parser.add_argument("--ph", type=float, default=7.0, help="pH for hydrogen addition (default: 7.0)")

    parser.add_argument("--forcefield", default="amber14-all.xml", help="Force field to use (default: amber14-all.xml)")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    return args


def minimize_trajectory(args):
    """Minimize each frame in the trajectory"""
    # Load trajectory
    print(f"Loading trajectory: {args.input}")
    traj = md.load(args.input)
    print(f"Loaded trajectory with {traj.n_frames} frames")

    # Convert to OpenMM topology
    topology = traj.topology.to_openmm()

    # Convert positions (list of frames)
    all_positions = [frame * unit.nanometers for frame in traj.xyz]
    if args.n_frames:
        all_positions = all_positions[: args.n_frames]

    # Create force field
    print(f"Using force field: {args.forcefield}")
    forcefield = app.ForceField(args.forcefield)

    all_minimized_positions = []
    for positions in tqdm.tqdm(all_positions, total=len(all_positions), desc="Minimizing structures"):
        # Add hydrogens
        modeller = app.Modeller(topology, positions)
        modeller.addHydrogens(pH=args.ph)

        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        # Create integrator
        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)

        # Create simulation object
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Minimize energy
        simulation.minimizeEnergy(maxIterations=args.max_iterations)

        # Save minimized structure
        minimized_positions = simulation.context.getState(getPositions=True).getPositions()
        all_minimized_positions.append(minimized_positions)

    # Save minimized trajectory
    print(f"Saving minimized trajectory to: {args.output}")
    minimized_traj = md.Trajectory(
        [pos.value_in_unit(unit.nanometers) for pos in all_minimized_positions],
        topology=md.Topology.from_openmm(modeller.topology),
    )
    minimized_traj.save(args.output)


def main():
    # Parse command line arguments
    args = parse_args()

    # Run minimization
    minimize_trajectory(args)

    print("Done!")


if __name__ == "__main__":
    main()
