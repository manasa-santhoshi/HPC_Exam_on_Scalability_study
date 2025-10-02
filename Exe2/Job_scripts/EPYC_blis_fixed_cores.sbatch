#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="EPYC_blis_fixed_cores"
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="EPYC_blis_fixedcores_summary.out"

echo "Running BLIS on EPYC..."


# Paths
sbatch_path=$(pwd)
blis_path=/u/dssc/msmaddi/blis_install
result_base=$HOME/HPC/Exe2/fixed_cores/EPYC


# Make sure BLIS library is found
export LD_LIBRARY_PATH=$blis_path/lib:$LD_LIBRARY_PATH

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
    else
        result_path="$result_base/spread_cores"
    fi

    mkdir -p $result_path
    echo "Running BLIS ($alloc)... results in $result_path"

    # Compile BLIS executables
    cd $sbatch_path
    make double data_folder=${result_path} blis_root=${blis_path}
    make float data_folder=${result_path} blis_root=${blis_path}
    echo "BLIS files compiled"

    # Move to result directory
    cd $result_path

    # Run all BLIS executables in a single srun to avoid socket timeouts
    srun -n1 --cpus-per-task=$ncores bash -c "
for size in \$(seq 2000 500 20000); do
    for iter in \$(seq 1 1 5); do
        $sbatch_path/../results/gemm_blis_d.exe \$size \$size \$size
        $sbatch_path/../results/gemm_blis_f.exe \$size \$size \$size
    done
done
    "

    echo "Execution finished for $alloc allocation"
done


