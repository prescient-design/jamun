#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=to_all_atom
#SBATCH --output=logs/%j_to_all_atom.log
#SBATCH --error=logs/%j_to_all_atom.err
#SBATCH --array=0-600

# Directory containing all input directories
BASE_INPUT_DIR="$1"
# Directory to store all frames
BASE_FRAMES_DIR="$2"
# Directory to store output
BASE_OUTPUT_DIR="$3"
# Maximum number of frames per directory
N_FRAMES="$4"

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
    
    # Run frame extraction script
    python scripts/IDRome/extract_frames.py --input-dir "${BASE_INPUT_DIR}" --name "${NAME}" --output-dir "${BASE_FRAMES_DIR}"

    # Process all frames in the directory
    for FRAME in $(seq 0 $((N_FRAMES-1))); do
        echo "Processing frame ${FRAME} in directory ${NAME}"
        
        # Check if the file exists
        FRAME_PATH="${BASE_FRAMES_DIR}/${NAME}/${FRAME}.pdb"
        if [ ! -f "${FRAME_PATH}" ]; then
            echo "Frame ${FRAME} does not exist in ${NAME}. Skipping."
            continue
        fi

        # Check if the final file already exists        
        PULCHRA_FRAME_PATH="${BASE_FRAMES_DIR}/${NAME}/${FRAME}.rebuilt.pdb"
        FINAL_PATH="${BASE_OUTPUT_DIR}/${NAME}/${FRAME}.pdb"

        if [ -f "${FINAL_PATH}" ]; then
            echo "Final file ${FINAL_PATH} already exists. Skipping."
            continue
        fi

        # Run PULCHRA
        echo "Running PULCHRA..."
        pulchra "${FRAME_PATH}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Error: PULCHRA failed on frame ${FRAME} in directory ${NAME}"
            continue
        fi
        
        # Move the rebuilt PDB to final location
        echo "Moving result to final location..."
        mv "${PULCHRA_FRAME_PATH}" "${FINAL_PATH}"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to move rebuilt PDB to final location"
            continue
        fi
        
        echo "Successfully processed frame ${FRAME} in directory ${NAME}"
    done

    echo "Completed processing directory ${NAME}"
done

exit 0