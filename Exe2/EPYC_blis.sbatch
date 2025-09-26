#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="blis_fixed_cores"
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="EPYC_summary.out"

echo "Loading modules..."
module load openBLAS/0.3.26-omp
echo "Modules loaded"

# Paths
sbatch_path=$(pwd)
blis_path=/u/dssc/msmaddi/blis_install
result_base=./fixed_cores
cd $sbatch_path

# Set OpenMP settings
ncores=64
alloc=spread       # change to 'close' if needed
export OMP_PLACES=cores
export OMP_PROC_BIND=$alloc
export OMP_NUM_THREADS=$ncores

# Determine result path
if [ "$alloc" = "close" ]; then
    result_path="$result_base/closed_cores"
else
    result_path="$result_base/spread_cores"
fi
mkdir -p $result_path
echo "Results will be stored in $result_path"

# Run executables
cd $result_path
for size in $(seq 2000 500 20000); do
    for iter in $(seq 1 1 5); do
        srun -n1 --cpus-per-task=$ncores $sbatch_path/../results/gemm_blis_d.exe $size $size $size
        srun -n1 --cpus-per-task=$ncores $sbatch_path/../results/gemm_blis_f.exe $size $size $size
    done
done

echo "Execution finished"
