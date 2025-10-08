#!/bin/bash
#SBATCH --job-name="OpenMP_scal"
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output="summary.out"

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

# Test all three matrix sizes: 10k, 15k, 20k
for matrix_dim in 8000 10000 12000; do
    echo "Initialising playground for ${matrix_dim}x${matrix_dim}"
    mpirun -np 1 ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    # OpenMP scalability: 1 to 12 threads
    for n_threads in $(seq 1 12); do
        export OMP_NUM_THREADS=$n_threads
        # Optional: repeat 5 times for averaging (as in her EPYC script)
        for rep in $(seq 1 5); do
            mpirun -np 1 --map-by socket ./$EXEC -r -f "playground_${matrix_dim}.pgm" -e 1 -n 50 -s 50
        done
    done
done

rm -f $EXEC playground_*.pgm
