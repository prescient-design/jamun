#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=combine_frames
#SBATCH --output=logs/%j_combine_frames.log
#SBATCH --error=logs/%j_combine_frames.err
#SBATCH --array=0-1

# Directory containing all input directories
BASE_INPUT_DIR="$1"
# Directory containing all original coarse-grained directories
BASE_ORIGINAL_DIR="$2"
# Directory to store output
BASE_OUTPUT_DIR="$3"

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

    # Check if the input frame exists
    if [ ! -f "${BASE_INPUT_DIR}/${NAME}/0_minimized_protein_0.pdb" ]; then
        echo "Input frame 0 does not exist in ${NAME}. Skipping."
        continue
    fi

    python scripts/IDRome/combine_frames.py \
        --name "${NAME}" \
        --input-dir "${BASE_INPUT_DIR}" \
        --original-traj-dir "${BASE_ORIGINAL_DIR}" \
        --output-dir "${BASE_OUTPUT_DIR}"

    echo "Completed processing directory ${NAME}"
done

exit 0
