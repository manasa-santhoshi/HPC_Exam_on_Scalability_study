#!/bin/bash
#SBATCH --job-name="OpenMP_Scaling_THIN"
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output="openmp_thin.out"

# --- Load environment ---
module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

# --- Compilation ---
echo "Compiling program..."
mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

# --- Matrix sizes to test ---
for matrix_dim in 4000 6000 8000; do
    echo "=== Initialising ${matrix_dim}x${matrix_dim} playground ==="
    mpirun -np 1 ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    echo "=== Running thread scaling for ${matrix_dim}x${matrix_dim} ==="
    # Loop over thread counts 1 → 12
    for n_threads in $(seq 1 12); do
        export OMP_NUM_THREADS=$n_threads
        echo "--- OMP_NUM_THREADS = $OMP_NUM_THREADS ---"

        # Run each config 5 times for averaging
        for rep in $(seq 1 5); do
            mpirun -np 1 \
                   --map-by socket --bind-to core --report-bindings \
                   -x OMP_NUM_THREADS \
                   ./$EXEC -r -f "playground_${matrix_dim}.pgm" -e 1 -n 50 -s 50
        done
    done
done

# --- Cleanup (optional) ---
rm -f $EXEC playground_*.pgm

echo "✅ THIN OpenMP scaling test completed."
