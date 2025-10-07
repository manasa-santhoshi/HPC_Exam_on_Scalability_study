#!/bin/bash
#SBATCH --job-name="Strong_scal"
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output="strong_epyc_summary.out"

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

# Loop over matrix sizes: 10k, 15k, 20k
for matrix_dim in 10000 15000 20000; do
    echo "Initialising playground for ${matrix_dim}x${matrix_dim}"
    mpirun -np 8 --map-by socket ./$EXEC -i -x $matrix_dim -y $matrix_dim -f "playground_${matrix_dim}.pgm"

    # Strong scaling: 1 to 8 MPI processes
    for n_procs in $(seq 1 8); do
        export OMP_NUM_THREADS=64
        mpirun -np $n_procs --map-by socket ./$EXEC -r -f "playground_${matrix_dim}.pgm" -e 1 -n 50 -s 50
    done
done

rm -f $EXEC playground_*.pgm
