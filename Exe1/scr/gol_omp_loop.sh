#!/bin/bash
#SBATCH --job-name=gol_omp_loop
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 MPI task → uses 1 socket
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00           # Allow time for 64 runs


module load openMPI/5.0.5

K=15000
NSTEPS=500
GRID_FILE="init_15000.pgm"

# Set up OpenMP environment (will override OMP_NUM_THREADS per iteration)
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=false

# --- PHASE 1: Initialize playground (once) ---
echo "🔹 Initializing playground (k=$K)..."
mpirun --map-by socket ./gameoflife -i -k $K -f $GRID_FILE
if [ $? -ne 0 ]; then
    echo "❌ Initialization failed!"
    exit 1
fi

for (( t = 1; t <= 64; t++ )); do
    echo "🔹 Running with OMP_NUM_THREADS=$t ..."
 
    export OMP_NUM_THREADS=$t
    # Run and capture output (your code prints time to stdout)
    output=$(mpirun --map-by socket ./gameoflife -r -f $GRID_FILE -k $K -n $NSTEPS -e 1 -s 0 2>&1)

    # Extract time from your program's output (e.g., "✅ Completed 500 steps in 2.3456 seconds")
    #if [[ $output =~ "Completed.*in ([0-9]+\.[0-9]+) seconds" ]]; then
    #   time_sec=${BASH_REMATCH[1]}
    #    echo "1,$t,$time_sec" >> $CSV_FILE
    #    echo "   → Time: $time_sec s"
    #else
    #    echo "   ⚠️ Failed to parse time. Output: $output"
    #    echo "1,$t,-1" >> $CSV_FILE
    #fi
done


