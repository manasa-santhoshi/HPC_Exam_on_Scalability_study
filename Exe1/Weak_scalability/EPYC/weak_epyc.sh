#!/bin/bash
#SBATCH --job-name="Weak_EPYC"
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=weak_epyc_summary.out

module load openMPI/5.0.5

cd "$SLURM_SUBMIT_DIR"
PROG=../../Program
EXEC=gol.exe

mpicc -fopenmp -DTIME -o $EXEC $PROG/game_of_life.c $PROG/gol_lib.c -std=c99

for np in $(seq 1 8); do
    width=10000
    height=$((10000 * np))
    mpirun -np 8 ./$EXEC -i -x $width -y $height -f "playground_${height}.pgm"
    export OMP_NUM_THREADS=64
    mpirun -np $np ./$EXEC -r -f "playground_${height}.pgm" -e 1 -n 50 -s 50
done

rm -f $EXEC playground_*.pgm
