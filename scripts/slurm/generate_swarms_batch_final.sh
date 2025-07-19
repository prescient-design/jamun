#!/bin/bash

# FINAL CORRECTED: Master script to generate SLURM batch jobs for swarm generation
# Correct equilibration steps: 50k restrained, 10 unrestrained
# Sequential processing: equilibration + swarms per structure before moving to next

echo "🚀 FINAL CORRECTED Swarm Batch Generation Script"
echo "================================================"

# Configuration
STRUCTURES_PER_BATCH=20
INPUT_DIR="data/swarm_data/test"
OUTPUT_DIR="data/swarm_data/test/swarm_results" 
SCRIPT_DIR="scripts/slurm/batches_final"

# Equilibration settings (CORRECTED)
NVT_RESTRAINT_STEPS=50000    # 50k as requested
NPT_RESTRAINT_STEPS=50000    # 50k as requested  
NVT_EQUIL_STEPS=10           # 10 steps (not 10k!) as requested
NPT_EQUIL_STEPS=10           # 10 steps (not 10k!) as requested

# Swarm settings
NUM_SWARMS=5
SWARM_STEPS=500              # 1ps ÷ 2fs/step = 500 steps per swarm
SAVE_FREQUENCY=10

# Create batch script directory
mkdir -p "$SCRIPT_DIR"

# Get list of all PDB files
PDB_FILES=($(ls -1 "$INPUT_DIR"/*.pdb | sort))
TOTAL_STRUCTURES=${#PDB_FILES[@]}

echo "📊 Configuration:"
echo "  Total structures: $TOTAL_STRUCTURES"
echo "  Structures per batch: $STRUCTURES_PER_BATCH"  
echo "  Equilibration steps: NVT/NPT restrained=50k, unrestrained=10 (CORRECTED)"
echo "  Swarms: $NUM_SWARMS × 1ps ($SWARM_STEPS steps) per structure"
echo "  Workflow: Sequential (equil+swarms per structure)"
echo ""

# Calculate number of batches needed
NUM_BATCHES=$(( (TOTAL_STRUCTURES + STRUCTURES_PER_BATCH - 1) / STRUCTURES_PER_BATCH ))

echo "📝 Generating $NUM_BATCHES FINAL CORRECTED batch scripts..."

for ((batch=1; batch<=NUM_BATCHES; batch++)); do
    # Calculate structure range for this batch
    start_idx=$(( (batch - 1) * STRUCTURES_PER_BATCH ))
    end_idx=$(( start_idx + STRUCTURES_PER_BATCH - 1 ))
    
    # Don't exceed total number of structures
    if [ $end_idx -ge $TOTAL_STRUCTURES ]; then
        end_idx=$(( TOTAL_STRUCTURES - 1 ))
    fi
    
    structures_in_batch=$(( end_idx - start_idx + 1 ))
    
    echo "  Batch $batch: global indices $start_idx-$end_idx ($structures_in_batch structures)"
    
    # Create SLURM script for this batch
    script_file="$SCRIPT_DIR/swarms_batch_${batch}_final.sh"
    
    cat > "$script_file" << EOF
#!/bin/bash
#SBATCH --job-name=swarms_batch_${batch}_final
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=swarms_batch_${batch}_final_%j.out
#SBATCH --error=swarms_batch_${batch}_final_%j.err

echo "SLURM_JOB_ID = \$SLURM_JOB_ID"
echo "hostname = \$(hostname)"
echo "Starting FINAL CORRECTED swarm batch ${batch} on GPU..."
echo ""

# Print GPU info
nvidia-smi

# Activate conda environment
echo "Activating conda environment..."
source /homefs/home/vanib/miniforge3/etc/profile.d/conda.sh
conda activate jamun
echo "Python path: \$(which python)"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo ""

# Change to working directory
cd /homefs/home/vanib/jamun

# Create full PDB file list
ALL_PDB_FILES=(
EOF

    # Add ALL PDB files to each script (needed for structure index validation)
    for ((i=0; i<TOTAL_STRUCTURES; i++)); do
        echo "    \"${PDB_FILES[$i]}\"" >> "$script_file"
    done
    
    cat >> "$script_file" << EOF
)

echo "🧬 Processing structures with GLOBAL indices $start_idx to $end_idx:"
echo "Using full PDB list of \${#ALL_PDB_FILES[@]} files for proper indexing"
echo "Sequential workflow: equilibration + swarms per structure"
echo ""

# Process each structure in this batch (SEQUENTIAL: equil+swarms per structure)
EOF

    # Add individual structure processing (SEQUENTIAL)
    for ((global_idx=start_idx; global_idx<=end_idx; global_idx++)); do
        pdb_file="${PDB_FILES[$global_idx]}"
        cat >> "$script_file" << EOF

echo "⚖️  Processing structure $global_idx: \$(basename "${pdb_file}")"
echo "============================================================"

# SINGLE COMMAND: Do both equilibration AND swarms for this structure
echo "🔄 Processing structure $global_idx: equilibration + $NUM_SWARMS × 1ps swarms..."
python scripts/generate_data/generate_swarms.py \\
    --input-pdbs "\${ALL_PDB_FILES[@]}" \\
    --output-dir "$OUTPUT_DIR" \\
    --single-structure \\
    --structure-index $global_idx \\
    --nvt-restraint-steps $NVT_RESTRAINT_STEPS \\
    --npt-restraint-steps $NPT_RESTRAINT_STEPS \\
    --nvt-equil-steps $NVT_EQUIL_STEPS \\
    --npt-equil-steps $NPT_EQUIL_STEPS \\
    --num-swarms $NUM_SWARMS \\
    --swarm-steps $SWARM_STEPS \\
    --save-frequency $SAVE_FREQUENCY \\
    --save-intermediate-files

if [ \$? -ne 0 ]; then
    echo "❌ Processing failed for structure $global_idx"
    exit 1
fi

echo "✅ COMPLETED structure $global_idx: \$(basename "${pdb_file}") (equilibration + $NUM_SWARMS swarms)"
echo ""
EOF
    done
    
    cat >> "$script_file" << EOF

# Summary
echo "📈 BATCH ${batch} SUMMARY"
echo "======================"
echo "  Global structure indices: $start_idx to $end_idx"
echo "  Structures processed: $structures_in_batch"
echo "  Swarms per structure: $NUM_SWARMS"
echo "  Total swarms generated: $(( structures_in_batch * NUM_SWARMS ))"
echo "  Swarm duration: 1ps each"
echo "  Equilibration: 50k restrained, 10 unrestrained steps"
echo "  Workflow: Sequential (equil+swarms per structure)"
echo ""
echo "🎉 Batch ${batch} completed successfully!"
EOF

    chmod +x "$script_file"
done

echo ""
echo "✅ Generated $NUM_BATCHES FINAL CORRECTED batch scripts in $SCRIPT_DIR/"
echo ""
echo "📋 To submit jobs:"
echo "  # Submit first batch only (for testing):"
echo "  sbatch $SCRIPT_DIR/swarms_batch_1_final.sh"
echo ""
echo "  # After verification, submit remaining batches:"
echo "  for i in {2..$NUM_BATCHES}; do"
echo "    sbatch $SCRIPT_DIR/swarms_batch_\${i}_final.sh"
echo "  done"
echo ""
echo "🔧 FINAL CORRECTIONS APPLIED:"
echo "  ✅ Correct equilibration steps: 50k restrained, 10 unrestrained"
echo "  ✅ Sequential workflow: equilibration + swarms per structure"
echo "  ✅ Passes all PDB files for proper structure index validation"
echo "  ✅ Each structure gets unique AA_XXX directory" 