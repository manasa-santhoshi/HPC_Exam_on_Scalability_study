#!/bin/bash
#SBATCH --job-name="OpenMP_THIN"
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=openmp_thin_summary.out

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

MATRIX=20000
./$EXEC -i -x $MATRIX -y $MATRIX -f "playground_${MATRIX}.pgm"

for threads in $(seq 1 12); do
    export OMP_NUM_THREADS=$threads
    ./$EXEC -r -f "playground_${MATRIX}.pgm" -e 1 -n 50 -s 50
done

rm -f $EXEC playground_*.pgm
