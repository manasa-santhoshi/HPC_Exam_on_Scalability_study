#!/bin/bash
#SBATCH --job-name="OpenMP_Scaling_EPYC"
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output="openmp_epyc.out"

# --- Load environment ---
module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

# --- Compilation ---
echo "Compiling program..."
mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

# --- Matrix sizes to test ---
for matrix_dim in 8000 10000 12000; do
    echo "=== Initialising ${matrix_dim}x${matrix_dim} playground ==="
    mpirun -np 1 ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    echo "=== Running thread scaling for ${matrix_dim}x${matrix_dim} ==="
    # Loop over thread counts 1 → 64
    for n_threads in $(seq 1 64); do
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

echo "✅ EPYC OpenMP scaling test completed."
