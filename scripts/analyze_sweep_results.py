#!/usr/bin/env python3
"""
Script to analyze results from the delta-friction parameter sweep.
"""

import numpy as np
import math
import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_parameter_grid():
    """Calculate the parameter grid used in the sweep."""
    sigma = 0.04
    
    # Delta values
    delta_min = sigma / math.sqrt(5)
    delta_max = math.sqrt(5) * sigma
    deltas = np.linspace(delta_min, delta_max, 5)
    
    # Friction values
    linear_points = np.linspace(0.01, 0.99, 5)
    frictions = [-math.log(p) for p in linear_points]
    
    return deltas, frictions, sigma

def fetch_sweep_results(project_name="sule-shashank/jamun"):
    """Fetch results from wandb for the parameter sweep."""
    api = wandb.Api()
    
    # Get runs with the sweep tag
    runs = api.runs(project_name, filters={"tags": {"$in": ["sweep", "delta_friction"]}})
    
    results = []
    for run in runs:
        if run.state == "finished":
            # Extract parameters from tags or config
            delta = None
            friction = None
            
            # Try to extract from tags first
            for tag in run.tags:
                if tag.startswith("delta_"):
                    try:
                        delta = float(tag.replace("delta_", ""))
                    except ValueError:
                        pass
                elif tag.startswith("friction_"):
                    try:
                        friction = float(tag.replace("friction_", ""))
                    except ValueError:
                        pass
            
            # Try to extract from config if not found in tags
            if delta is None:
                delta = run.config.get("delta")
            if friction is None:
                friction = run.config.get("friction")
            
            if delta is not None and friction is not None:
                # Get metrics (adjust these based on what metrics you're interested in)
                metrics = {}
                if run.summary:
                    # Add metrics you want to analyze
                    for key in ["sampling_time", "chemical_validity", "ramachandran_score"]:
                        if key in run.summary:
                            metrics[key] = run.summary[key]
                
                results.append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "delta": delta,
                    "friction": friction,
                    **metrics
                })
    
    return pd.DataFrame(results)

def create_heatmaps(df, deltas, frictions):
    """Create heatmaps for each metric."""
    if df.empty:
        print("No results found to plot.")
        return
    
    # Get metric columns (exclude parameter and metadata columns)
    metric_cols = [col for col in df.columns if col not in ["run_id", "run_name", "delta", "friction"]]
    
    if not metric_cols:
        print("No metrics found in the data.")
        return
    
    # Create a figure with subplots for each metric
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metric_cols):
        # Create a pivot table for the heatmap
        pivot_data = df.pivot(index="friction", columns="delta", values=metric)
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            ax=axes[i], 
            annot=True, 
            fmt=".4f", 
            cmap="viridis",
            cbar_kws={"label": metric}
        )
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        axes[i].set_xlabel("Delta")
        axes[i].set_ylabel("Friction")
    
    plt.tight_layout()
    plt.savefig("sweep_results_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

def print_summary(df, deltas, frictions):
    """Print a summary of the sweep results."""
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    
    print(f"Parameter grid:")
    print(f"  Deltas: {len(deltas)} values from {deltas[0]:.6f} to {deltas[-1]:.6f}")
    print(f"  Frictions: {len(frictions)} values from {frictions[0]:.6f} to {frictions[-1]:.6f}")
    print(f"  Total combinations: {len(deltas) * len(frictions)}")
    
    print(f"\nResults found: {len(df)} / {len(deltas) * len(frictions)}")
    
    if not df.empty:
        print(f"\nMetrics available:")
        metric_cols = [col for col in df.columns if col not in ["run_id", "run_name", "delta", "friction"]]
        for metric in metric_cols:
            print(f"  - {metric}")
        
        print(f"\nBest performing combinations:")
        for metric in metric_cols:
            if metric in df.columns:
                if "time" in metric.lower():
                    # For time metrics, lower is better
                    best_idx = df[metric].idxmin()
                    print(f"  {metric} (lowest): delta={df.loc[best_idx, 'delta']:.6f}, friction={df.loc[best_idx, 'friction']:.6f}, value={df.loc[best_idx, metric]:.6f}")
                else:
                    # For other metrics, higher is usually better
                    best_idx = df[metric].idxmax()
                    print(f"  {metric} (highest): delta={df.loc[best_idx, 'delta']:.6f}, friction={df.loc[best_idx, 'friction']:.6f}, value={df.loc[best_idx, metric]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze delta-friction parameter sweep results")
    parser.add_argument("--project", default="sule-shashank/jamun", help="Wandb project name")
    parser.add_argument("--plot", action="store_true", help="Create heatmap plots")
    parser.add_argument("--save-csv", help="Save results to CSV file")
    
    args = parser.parse_args()
    
    # Calculate parameter grid
    deltas, frictions, sigma = calculate_parameter_grid()
    
    print("Fetching results from wandb...")
    df = fetch_sweep_results(args.project)
    
    # Print summary
    print_summary(df, deltas, frictions)
    
    # Save to CSV if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nResults saved to {args.save_csv}")
    
    # Create plots if requested
    if args.plot and not df.empty:
        create_heatmaps(df, deltas, frictions)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
