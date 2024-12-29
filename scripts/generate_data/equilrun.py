import os
import sys


import openmm_utils as op

protein = sys.argv[1]
os.chdir(protein)

temp = 300  # 300K temperature

pressure = 1.0


positions, topology = op.fix_pdb("init.pdb")  # Fixes pdb with misiing atoms and terminal atoms

# ff = ForceField('charmm36.xml', 'charmm36/water.xml')         # Creates forcefield
ff = ForceField("amber99sbildn.xml", "tip3p.xml")  # Creates forcefield - for DDR1 kinase

positions, topology = op.add_hydrogen(positions, topology, ff)  # Adds hydrogen according to the forcefield
positions, topology = op.solvate_me(
    positions, topology, ff, True, 1, "tip3p"
)  # (Default) neutralizes the charge with Na+ and Cl- ions

# Create simulation
simulation = op.get_system_with_Langevin_integrator(topology, ff, temp, 0.002)


# Perform equilibration

# Adding restrain
simulation = op.add_position_restraints(positions, topology, simulation, 10)  # 10 KJ/(mol.A^2) contrain

# Minimize
positions, simulation = op.minimize_energy(positions, simulation)

# NVT - restrain
positions, velocities, simulation = op.run_MD(
    positions, simulation, 150000, ens="NVT", run_type="restrainedNVT"
)  # No '_' in run_type

positions, velocities, simulation = op.run_MD(
    positions, simulation, 150000, ens="NPT", run_type="restrainedNPT", velocities=velocities, cont=True
)

# Remove restrain
simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces() - 1)

# NVT - no restrain
positions, velocities, simulation = op.run_MD(
    positions, simulation, 250000, ens="NVT", run_type="equilNVT", velocities=velocities, cont=True
)


# NPT - no restrain
positions, velocities, simulation = op.run_MD(
    positions, simulation, 10000000, ens="NPT", run_type="equilNPT", velocities=velocities, cont=True
)
