#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="EPYC_openblas_close_spread"
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="EPYC_openblas_fixedcores_summary.out"

echo "Loading OpenBLAS module..."
module load openBLAS/0.3.26-omp
echo "OpenBLAS module loaded"

# Paths
sbatch_path=$(pwd)
result_base=$HOME/HPC/Exe2/fixed_cores/EPYC

# OpenMP settings
ncores=64
export OMP_NUM_THREADS=$ncores
export OMP_PLACES=cores

# Loop over allocations
for alloc in close spread
do
    export OMP_PROC_BIND=$alloc

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

    # Run all OpenBLAS executables in a single srun per allocation
    srun -n1 --cpus-per-task=$ncores bash -c "
for size in \$(seq 2000 500 20000); do
    for iter in \$(seq 1 1 5); do
        $sbatch_path/../results/gemm_openblas_d.exe \$size \$size \$size | grep '^[0-9]' >> $csv_d
        $sbatch_path/../results/gemm_openblas_f.exe \$size \$size \$size | grep '^[0-9]' >> $csv_f
    done
done
"

    echo "Execution finished for $alloc allocation"
done

echo "All OpenBLAS runs completed"
