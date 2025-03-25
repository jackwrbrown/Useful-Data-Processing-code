#!/bin/bash

#SBATCH --qos bbdefault
#SBATCH --array=0-95
#SBATCH --nodes 1
#SBATCH --mem=64G  # Memory per job (adjust as needed)
#SBATCH --ntasks 1
#SBATCH --account=themendr-j-brown
#SBATCH --time 02:00:00
#SBATCH --output=DT_GNSS_processing_info/job_%A_%a.out   # Save the .out file to DT_GNSS_processing_info
#SBATCH --error=DT_GNSS_processing_info/job_%A_%a.stats  # Save the .stats (error) file to DT_GNSS_processing_info

# Set the path to the directory where your script is located
cd /rds/homes/j/jxb1605/themendr-j-brown/DRT_Feb_8_2025_Data/Analysing_global_DT_GNSS_data/

# Create the output directory if it doesn't exist
mkdir -p DT_GNSS_processing_info

# Load necessary modules
module purge
module load bluebear
module load bear-apps/2023a
module load netcdf4-python/1.6.4-intel-2023a


# Set OpenMP threads (if necessary for parallelization)
export OMP_NUM_THREADS=1

# Use the SLURM_ARRAY_TASK_ID to dynamically compute the start_time, change periods to 96 for full day (change the array to 0-95)
START_TIME=$(python -c "import pandas as pd; intervals = pd.date_range('2025-02-08 00:00:00', periods=96, freq='15T'); print(intervals[$SLURM_ARRAY_TASK_ID].strftime('%Y-%m-%d %H:%M:%S'))")

# Run the Python script with the dynamically generated start_time
python -u DT_GNSS_data_to_AENeAS_format.py --start_time "$START_TIME"