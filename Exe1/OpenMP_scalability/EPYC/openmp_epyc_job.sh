#!/bin/bash
#SBATCH --job-name="OpenMP_scal"
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output="summary.out"

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

for matrix_dim in 10000 15000 20000; do
    mpirun -np 1 ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    # Her EPYC script used threads 24→64, but report shows full range
    # To match Figure 1, use full range: 1→64
    for n_threads in $(seq 1 64); do
        export OMP_NUM_THREADS=$n_threads
        for rep in $(seq 1 5); do
            mpirun -np 1 --map-by socket ./$EXEC -r -f "playground_${matrix_dim}.pgm" -e 1 -n 50 -s 50
        done
    done
done

rm -f $EXEC playground_*.pgm
