#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=96
#SBATCH --job-name=relax_structures
#SBATCH --output=logs/%j_relax_structures.log
#SBATCH --error=logs/%j_relax_structures.err
#SBATCH --array=0-1

# Directory containing all input directories
BASE_INPUT_DIR="$1"
# Directory to store output
BASE_OUTPUT_DIR="$2"
# Maximum number of frames per directory
N_FRAMES="$3"

eval "$(conda shell.bash hook)"
conda activate jamun

# Get list of all directories and store in an array
# You can use a file with directory names or generate the list dynamically
DIRECTORIES=($(ls -d ${BASE_INPUT_DIR}/*/ | sort | xargs -n 1 basename))

# Each job processes 50 directories
START_IDX=$((SLURM_ARRAY_TASK_ID * 50))
END_IDX=$(( (SLURM_ARRAY_TASK_ID + 1) * 50 - 1 ))

for DIR_INDEX in $(seq ${START_IDX} ${END_IDX}); do
    NAME="${DIRECTORIES[${DIR_INDEX}]}"

    echo "Processing directory: ${NAME} (index: ${DIR_INDEX})"

    # Create output directory
    mkdir -p "${BASE_OUTPUT_DIR}/${NAME}"

    # Check if the last frame already exists
    if [ -f "${BASE_OUTPUT_DIR}/${NAME}/$((${N_FRAMES}-1)).pdb" ]; then
        echo "Final file already exists for ${NAME}. Skipping."
        continue
    fi
    
    # Process all frames in the directory
    for FRAME in $(seq 0 $((N_FRAMES-1))); do
        echo "Processing frame ${FRAME} in directory ${NAME}"

        # Check if the final file already exists        
        INPUT_PATH="${BASE_INPUT_DIR}/${NAME}/${FRAME}.pdb"
        FINAL_PATH="${BASE_OUTPUT_DIR}/${NAME}/${FRAME}_minimized_protein_0.pdb"

        if [ -f "${FINAL_PATH}" ]; then
            echo "Final file ${FINAL_PATH} already exists. Skipping."
            continue
        fi

        python scripts/generate_data/run_simulation.py "${INPUT_PATH}" \
        --output-dir "${BASE_OUTPUT_DIR}/${NAME}" \
        --energy-minimization-only \
        --energy-minimization-steps=1000
        
        echo "Successfully processed frame ${FRAME} in directory ${NAME}"
    done

    echo "Completed processing directory ${NAME}"
done

exit 0