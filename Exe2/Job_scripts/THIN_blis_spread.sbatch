#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="THIN_blis_fixed_cores_spread"
#SBATCH --partition=THIN
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="THIN_fixedcores_spread_summary.out"

echo "Loading modules..."
# Only needed if you use MKL/OpenBLAS as well, not strictly for BLIS
# module load openBLAS/0.3.26-omp
echo "Modules loaded"

# Paths
sbatch_path=$(pwd)
blis_path=/u/dssc/msmaddi/blis_install
result_base=./fixed_cores/THIN
cd $sbatch_path

# OpenMP settings
ncores=12
alloc=spread      
export OMP_PLACES=cores
export OMP_PROC_BIND=$alloc
export OMP_NUM_THREADS=$ncores

# Add BLIS shared library to runtime search path
export LD_LIBRARY_PATH=$blis_path/lib:$LD_LIBRARY_PATH

# Result path
if [ "$alloc" = "close" ]; then
    result_path="$result_base/close_cores"
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
