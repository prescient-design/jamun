import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from jamun.metrics._chemical_validity import check_bond_lengths

# a. Load trajectory and topology
traj_path_conditional = "/data2/sules/jamun-conditional-runs/outputs/sample/dev/runs/2025-08-13_20-22-36/sampler/ALA_ALA/predicted_samples/dcd/joined.dcd"
pdb_path_conditional = "/data2/sules/jamun-conditional-runs/outputs/sample/dev/runs/2025-08-13_20-22-36/sampler/ALA_ALA/topology.pdb"
traj_path_unconditional = "/data2/sules/jamun-conditional-runs//outputs/sample/dev/runs/2025-08-19_18-56-30/sampler/ALA_ALA/predicted_samples/dcd/joined.dcd"
pdb_path_unconditional = "/data2/sules/jamun-conditional-runs//outputs/sample/dev/runs/2025-08-19_18-56-30/sampler/ALA_ALA/topology.pdb"
md_traj_path = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.xtc"
md_pdb_path = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
# Load trajectory with topology
traj_conditional = md.load(traj_path_conditional, top=pdb_path_conditional)
traj_unconditional = md.load(traj_path_unconditional, top=pdb_path_unconditional)
md_traj = md.load(md_traj_path, top=md_pdb_path)
md_traj = md_traj[::28]
breakpoint()

# b. Check bond length issues
tolerance = 0.1  # 20% tolerance (commonly used value)
bond_length_issues_conditional = check_bond_lengths(traj_conditional, tolerance=tolerance)
bond_length_issues_unconditional = check_bond_lengths(traj_unconditional, tolerance=tolerance)
bond_length_issues_md = check_bond_lengths(md_traj, tolerance=tolerance)
breakpoint()
print(f"\nBond length analysis (tolerance: {tolerance*100}%):")
print(f"Number of frames analyzed: {len(bond_length_issues_conditional)}")

# Convert to numpy array for easier analysis
issues_array_conditional = np.array(bond_length_issues_conditional)
total_issues_conditional = np.sum(issues_array_conditional)
cumulants_conditional = np.array([np.sum(issues_array_conditional[:i])/np.sum(issues_array_conditional) for i in range(issues_array_conditional.shape[0])])

issues_array_unconditional = np.array(bond_length_issues_unconditional)
total_issues_unconditional = np.sum(issues_array_unconditional)
cumulants_unconditional = np.array([np.sum(issues_array_unconditional[:i])/np.sum(issues_array_unconditional) for i in range(issues_array_unconditional.shape[0])])

issues_array_md = np.array(bond_length_issues_md)
total_issues_md = np.sum(issues_array_md)
cumulants_md = np.array([np.sum(issues_array_md[:i])/np.sum(issues_array_md) for i in range(issues_array_md.shape[0])])

breakpoint()
# Create histogram
plt.figure(figsize=(12, 8))

# Main histogram
plt.subplot(1, 2, 1)
plt.hist(issues_array_conditional, bins=10, alpha=0.7, range=(0.0, 1.0), edgecolor='black', color='blue', label=f'KALA-JAMUN')
plt.hist(issues_array_unconditional, bins=10, alpha=0.7, range=(0.0, 1.0), edgecolor='black', color='red', label=f'JAMUN')
plt.hist(issues_array_md, bins=10, alpha=0.7, range=(0.0, 1.0), edgecolor='black', color='green', label=f'Reference MD Trajectory issues')
plt.legend()
plt.xlabel('Fraction of Bonds with Issues', fontsize=14)
plt.ylabel('Number of Frames', fontsize=14)
plt.ylim(0, 5.0e4)
plt.title(f'Distribution of Bond Length Issues\n(Tolerance: {tolerance*100}%)', fontsize=14)
plt.grid(True, alpha=0.3)

# Time series plot
plt.subplot(1, 2, 2)
plt.plot(np.linspace(0,1,issues_array_conditional.shape[0]), cumulants_conditional, alpha=0.5, color='blue', label='KALA-JAMUN', linewidth=5)
plt.plot(np.linspace(0,1,issues_array_unconditional.shape[0]), cumulants_unconditional, alpha=0.5, color='red', label='JAMUN', linewidth=5)
plt.plot(np.linspace(0,1,issues_array_md.shape[0]), cumulants_md, alpha=0.5, color='green', label='Reference MD Trajectory', linewidth=5)
plt.legend()
plt.xlabel('Prop. of trajectory length', fontsize=14)
plt.ylabel('Fraction of issues arising', fontsize=14)
plt.title('Bond Issues Over Time', fontsize=14)
plt.grid(True, alpha=0.3)

plt.savefig("bond_length_issues_conditional_traj.png")