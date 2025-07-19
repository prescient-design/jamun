#!/bin/bash

# Helper script for parallelizing swarm generation
# This script demonstrates different approaches for running generate_swarms.py in parallel

# Default parameters
INPUT_FOLDER=""
INPUT_PDBS=""
OUTPUT_DIR=""
NUM_SWARMS=10
SWARM_STEPS=10000
SAVE_FREQUENCY=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --input-pdbs)
            shift
            INPUT_PDBS=""
            while [[ $# -gt 0 && $1 != --* ]]; do
                INPUT_PDBS="$INPUT_PDBS $1"
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-swarms)
            NUM_SWARMS="$2"
            shift 2
            ;;
        --swarm-steps)
            SWARM_STEPS="$2"
            shift 2
            ;;
        --save-frequency)
            SAVE_FREQUENCY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-folder DIR     Folder containing PDB files"
            echo "  --input-pdbs FILE...   List of PDB files"
            echo "  --output-dir DIR       Output directory"
            echo "  --num-swarms N         Number of swarms per structure (default: 10)"
            echo "  --swarm-steps N        Steps per swarm (default: 10000)"
            echo "  --save-frequency N     Save frequency (default: 10)"
            echo "  --help                 Show this help"
            echo ""
            echo "Examples:"
            echo "  # Using SLURM job arrays:"
            echo "  $0 --input-folder /path/to/pdbs --output-dir results"
            echo ""
            echo "  # Using GNU parallel:"
            echo "  $0 --input-pdbs *.pdb --output-dir results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

if [[ -z "$INPUT_FOLDER" && -z "$INPUT_PDBS" ]]; then
    echo "Error: Either --input-folder or --input-pdbs is required"
    exit 1
fi

# Get list of PDB files
if [[ -n "$INPUT_FOLDER" ]]; then
    PDB_FILES=($(find "$INPUT_FOLDER" -name "*.pdb" | sort))
    echo "Found ${#PDB_FILES[@]} PDB files in $INPUT_FOLDER"
else
    PDB_FILES=($INPUT_PDBS)
    echo "Processing ${#PDB_FILES[@]} specified PDB files"
fi

if [[ ${#PDB_FILES[@]} -eq 0 ]]; then
    echo "Error: No PDB files found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Swarm Generation Parallelization Helper"
echo "=========================================="
echo "PDB files to process: ${#PDB_FILES[@]}"
echo "Output directory: $OUTPUT_DIR"
echo "Swarms per structure: $NUM_SWARMS"
echo "Steps per swarm: $SWARM_STEPS"
echo "Save frequency: $SAVE_FREQUENCY"
echo ""

# Method 1: SLURM Job Array
echo "=== SLURM Job Array Approach ==="
echo "To submit as a SLURM job array, create a file 'submit_swarms.sh':"
echo ""
cat << 'EOF'
#!/bin/bash
#SBATCH --job-name=swarms
#SBATCH --array=0-NUM_STRUCTURES-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=swarms_%A_%a.out
#SBATCH --error=swarms_%A_%a.err

# Get PDB file for this array job
PDB_FILES=(PDB_FILE_LIST)
PDB_FILE=${PDB_FILES[$SLURM_ARRAY_TASK_ID]}

# Run swarm generation for single structure
python scripts/generate_data/generate_swarms.py \
    --input-pdbs "$PDB_FILE" \
    --output-dir OUTPUT_DIR \
    --single-structure \
    --structure-index $SLURM_ARRAY_TASK_ID \
    --num-swarms NUM_SWARMS \
    --swarm-steps SWARM_STEPS \
    --save-frequency SAVE_FREQUENCY
EOF

# Create actual SLURM script
SLURM_SCRIPT="submit_swarms_$(date +%Y%m%d_%H%M%S).sh"
sed "s/NUM_STRUCTURES/$((${#PDB_FILES[@]}-1))/" << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=swarms
#SBATCH --array=0-NUM_STRUCTURES
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=swarms_%A_%a.out
#SBATCH --error=swarms_%A_%a.err

# Get PDB file for this array job
EOF

echo "PDB_FILES=(" >> "$SLURM_SCRIPT"
for pdb in "${PDB_FILES[@]}"; do
    echo "    \"$pdb\"" >> "$SLURM_SCRIPT"
done
echo ")" >> "$SLURM_SCRIPT"

cat << EOF >> "$SLURM_SCRIPT"
PDB_FILE=\${PDB_FILES[\$SLURM_ARRAY_TASK_ID]}

# Run swarm generation for single structure
python scripts/generate_data/generate_swarms.py \\
    --input-pdbs "\$PDB_FILE" \\
    --output-dir "$OUTPUT_DIR" \\
    --single-structure \\
    --structure-index \$SLURM_ARRAY_TASK_ID \\
    --num-swarms $NUM_SWARMS \\
    --swarm-steps $SWARM_STEPS \\
    --save-frequency $SAVE_FREQUENCY
EOF

echo "Created SLURM script: $SLURM_SCRIPT"
echo "To submit: sbatch $SLURM_SCRIPT"
echo ""

# Method 2: GNU Parallel
echo "=== GNU Parallel Approach ==="
echo "To run with GNU parallel:"

# Create parallel command file
PARALLEL_SCRIPT="run_swarms_parallel_$(date +%Y%m%d_%H%M%S).sh"
cat << EOF > "$PARALLEL_SCRIPT"
#!/bin/bash

# Function to process a single PDB file
process_pdb() {
    local pdb_file="\$1"
    local structure_idx="\$2"
    
    echo "Processing \$pdb_file (structure \$structure_idx)"
    
    python scripts/generate_data/generate_swarms.py \\
        --input-pdbs "\$pdb_file" \\
        --output-dir "$OUTPUT_DIR" \\
        --single-structure \\
        --structure-index "\$structure_idx" \\
        --num-swarms $NUM_SWARMS \\
        --swarm-steps $SWARM_STEPS \\
        --save-frequency $SAVE_FREQUENCY
}

export -f process_pdb

# Run in parallel (adjust -j for number of parallel jobs)
parallel -j 4 process_pdb {1} {#} ::: \\
EOF

for pdb in "${PDB_FILES[@]}"; do
    echo "    \"$pdb\" \\" >> "$PARALLEL_SCRIPT"
done

# Remove last backslash
sed -i '$ s/ \\$//' "$PARALLEL_SCRIPT"

chmod +x "$PARALLEL_SCRIPT"
echo "Created parallel script: $PARALLEL_SCRIPT"
echo "To run: ./$PARALLEL_SCRIPT"
echo ""

# Method 3: Simple Background Jobs
echo "=== Background Jobs Approach ==="
BACKGROUND_SCRIPT="run_swarms_background_$(date +%Y%m%d_%H%M%S).sh"
cat << EOF > "$BACKGROUND_SCRIPT"
#!/bin/bash

echo "Running swarm generation with background jobs..."

# Process each PDB file in background (limit concurrent jobs)
max_jobs=4  # Adjust based on your system
job_count=0

EOF

for i in "${!PDB_FILES[@]}"; do
    cat << EOF >> "$BACKGROUND_SCRIPT"
# Wait if we've hit the job limit
while [ \$(jobs -r | wc -l) -ge \$max_jobs ]; do
    sleep 1
done

echo "Starting structure $i: ${PDB_FILES[$i]}"
python scripts/generate_data/generate_swarms.py \\
    --input-pdbs "${PDB_FILES[$i]}" \\
    --output-dir "$OUTPUT_DIR" \\
    --single-structure \\
    --structure-index $i \\
    --num-swarms $NUM_SWARMS \\
    --swarm-steps $SWARM_STEPS \\
    --save-frequency $SAVE_FREQUENCY &

EOF
done

cat << 'EOF' >> "$BACKGROUND_SCRIPT"

# Wait for all background jobs to complete
echo "Waiting for all jobs to complete..."
wait

echo "All swarm generation jobs completed!"
EOF

chmod +x "$BACKGROUND_SCRIPT"
echo "Created background jobs script: $BACKGROUND_SCRIPT"
echo "To run: ./$BACKGROUND_SCRIPT"
echo ""

# Method 4: Single command (no parallelization)
echo "=== Single Process Approach ==="
echo "To run all structures in a single process:"
if [[ -n "$INPUT_FOLDER" ]]; then
    SINGLE_CMD="python scripts/generate_data/generate_swarms.py --input-folder \"$INPUT_FOLDER\""
else
    SINGLE_CMD="python scripts/generate_data/generate_swarms.py --input-pdbs"
    for pdb in "${PDB_FILES[@]}"; do
        SINGLE_CMD="$SINGLE_CMD \"$pdb\""
    done
fi
SINGLE_CMD="$SINGLE_CMD --output-dir \"$OUTPUT_DIR\" --num-swarms $NUM_SWARMS --swarm-steps $SWARM_STEPS --save-frequency $SAVE_FREQUENCY"

echo "$SINGLE_CMD"
echo ""

echo "=========================================="
echo "Choose the parallelization method that best fits your computing environment:"
echo "1. SLURM job arrays - Best for HPC clusters"
echo "2. GNU parallel - Good for multi-core workstations"  
echo "3. Background jobs - Simple shell-based parallelization"
echo "4. Single process - No parallelization, simplest approach"
echo "==========================================" 