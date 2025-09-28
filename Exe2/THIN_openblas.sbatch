#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="THIN_openblas_close_spread"
#SBATCH --partition=THIN
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="THIN_openblas_summary.out"

echo "Loading OpenBLAS module..."
module load openBLAS/0.3.29-omp
echo "OpenBLAS module loaded"

# Paths
sbatch_path=$(pwd)
result_base=$HOME/HPC/Exe2/fixed_cores/THIN

# Set OpenMP settings
ncores=12

for alloc in close spread
do
    export OMP_PLACES=cores
    export OMP_PROC_BIND=$alloc
    export OMP_NUM_THREADS=$ncores

    if [ "$alloc" = "close" ]; then
        result_path="$result_base/close_cores"
        csv_d="$result_path/openblas_d.csv"
        csv_f="$result_path/openblas_f.csv"
    else
        result_path="$result_base/spread_cores"
        csv_d="$result_path/openblas_d.csv"
        csv_f="$result_path/openblas_f.csv"
    fi

    mkdir -p $result_path
    echo "Running OpenBLAS ($alloc)... results in $result_path"

    cd $result_path
    # Run tests and append results to CSV
    for size in $(seq 2000 500 20000); do
        for iter in $(seq 1 1 5); do
            srun -n1 --cpus-per-task=$ncores $sbatch_path/../results/gemm_openblas_d.exe $size $size $size >> $csv_d
            srun -n1 --cpus-per-task=$ncores $sbatch_path/../results/gemm_openblas_f.exe $size $size $size >> $csv_f
        done
    done
done

echo "Execution finished"
