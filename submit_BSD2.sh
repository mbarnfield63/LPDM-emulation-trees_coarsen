#!/bin/bash

#SBATCH --partition=short,compute
#SBATCH --mem=150GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=met
#SBTACH --output=met
#SBATCH --time=12:00:00
#SBATCH --account=chem007981


# Activate python environment
echo "activate env"
source activate /user/home/eh19374/.conda/envs/trees_emulator

# Run train_emulator.py
cd /user/work/eh19374/LPDM-emulation-trees_TESTING/

# Start recording the start time
start_time=$(date +%s)
python train_emulator.py "BSD" "201[4-5]" --size 9 --input_size 14 --coarsen 2 --hours_back 6 --save_name "BSD_1415_coarsened2"

# End recording the end time
end_time=$(date +%s)

# Calculate and print the runtime
runtime=$((end_time - start_time))
echo "BSD Coarsen 2 runtime: $runtime seconds"