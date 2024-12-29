import os
import sys
import time

import mdtraj as md
import openmm_utils as op  # openmm functions

from openmm import CustomExternalForce, MonteCarloBarostat, PeriodicTorsionForce, Platform, XmlSerializer
from openmm.app import PME, CheckpointReporter, HBonds, Modeller, PDBFile, Simulation, StateDataReporter, ForceField
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

protein = sys.argv[1]
name = sys.argv[2]
os.chdir(protein)

temp = 300  # 300K temperature

pressure = 1.0


pdb = PDBFile("%s.pdb" % name)

topology = pdb.topology
positions = pdb.positions

ff = ForceField("amber99sbildn.xml", "tip3p.xml")
simulation = op.get_system_with_Langevin_integrator(topology, ff, temp, 0.002)

top = md.Topology.from_openmm(simulation.topology)
traj = md.load("%s.pdb" % name)


with open("%s.state" % name, "r") as f:
    state_xml = f.read()

state = XmlSerializer.deserialize(state_xml)
velocities = state.getVelocities()
# NPT - no restrain


# get time

start_time = time.time()
positions, velocities, simulation = op.run_MD(
    positions, simulation, 5000, ens="NPT", run_type="equilNPT", velocities=velocities, cont=True
)

# print time to "time.txt" file in seconds

time_file = open("time.txt", "w")
time_file.write("\n%s\n" % (time.time() - start_time))
time_file.close()
