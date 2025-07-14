#!/usr/bin/env python3
"""
Show all parameter combinations for the enhanced sampling sweep.
"""

# Define parameter arrays (same as in the bash script)
CONDITIONERS = ["PositionConditioner", "SelfConditioner"]
SIGMAS = [0.01, 0.04, 0.08, 0.1]
LAG_TIMES = [2, 5, 8]

print("Enhanced Sampling Training Sweep - Parameter Combinations")
print("=" * 60)
print(f"Total combinations: {len(CONDITIONERS)} × {len(SIGMAS)} × {len(LAG_TIMES)} = {len(CONDITIONERS) * len(SIGMAS) * len(LAG_TIMES)}")
print()
print(f"{'Task ID':<8} {'Conditioner':<18} {'Sigma':<8} {'Lag Time':<10}")
print("-" * 45)

task_id = 0
for cond_idx, conditioner in enumerate(CONDITIONERS):
    for sigma_idx, sigma in enumerate(SIGMAS):
        for lag_idx, lag_time in enumerate(LAG_TIMES):
            print(f"{task_id:<8} {conditioner:<18} {sigma:<8} {lag_time:<10}")
            task_id += 1

print()
print("To run the sweep:")
print("sbatch scripts/slurm/train_enhanced_sampling_sweep.sh") 