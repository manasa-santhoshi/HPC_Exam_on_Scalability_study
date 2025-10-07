#!/bin/bash
#SBATCH --job-name="Strong_EPYC"
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=strong_epyc_summary.out

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

for size in 10000 15000 20000; do
    mpirun -np 8 ./$EXEC -i -x $size -y $size -f "playground_${size}.pgm"
    for np in $(seq 1 8); do
        export OMP_NUM_THREADS=64
        mpirun -np $np ./$EXEC -r -f "playground_${size}.pgm" -e 1 -n 50 -s 50
    done
done

rm -f $EXEC playground_*.pgm
