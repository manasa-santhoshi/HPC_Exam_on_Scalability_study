#!/bin/bash
#SBATCH --job-name=gol_full
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=2              # 1 MPI task per socket
#SBATCH --cpus-per-task=64     # 12 OpenMP threads per task
#SBATCH --exclusive
#SBATCH --time=00:30:00


# Load required module
module load openMPI/5.0.5

# Set OpenMP environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=false

# --- PHASE 1: Initialize playground ---
echo "🔹 Phase 1: Initializing playground (k=1000)..."
mpirun --map-by socket ./gameoflife -i -k 1000 -f init.pgm

if [ $? -ne 0 ]; then
    echo "❌ Initialization failed!"
    exit 1
fi

# --- PHASE 2: Run simulation ---
echo "🔹 Phase 2: Running Game of Life for 100 steps..."
mpirun --map-by socket ./gameoflife -r -f init.pgm -k 1000 -n 100 -e 1 -s 10

if [ $? -ne 0 ]; then
    echo "❌ Simulation failed!"
    exit 1
fi

echo " Done: EPYC run completed successfully."
