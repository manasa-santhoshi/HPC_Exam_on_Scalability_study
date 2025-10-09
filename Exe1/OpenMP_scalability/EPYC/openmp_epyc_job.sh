#!/bin/bash
#SBATCH --job-name="OpenMP_Scaling_EPYC"
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output="openmp_epyc.out"

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

echo "Compiling program..."
mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

for matrix_dim in 8000 10000 12000; do
    echo "=== Initialising ${matrix_dim}x${matrix_dim} ==="
    mpirun -np 1 ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    echo "=== Testing thread scaling for ${matrix_dim} ==="
    for n_threads in 1 2 4 8 12 16 24 32 48 64; do
        export OMP_NUM_THREADS=$n_threads
        for rep in {1..3}; do
            mpirun -np 1 \
                   --map-by socket --bind-to core -x OMP_NUM_THREADS \
                   ./$EXEC -r -f "playground_${matrix_dim}.pgm" -e 1 -n 50 -s 50
        done
    done
done

rm -f $EXEC playground_*.pgm
echo "✅ EPYC OpenMP test completed."
