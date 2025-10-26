#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="fixedmatrix_epyc_blis"
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="EPYC_fixedmatrix_blis.out"

echo "Running Fixed Matrix test – BLIS only on EPYC"

# Paths
sbatch_path=$(pwd)
blis_path=/u/dssc/msmaddi/blis_install
export LD_LIBRARY_PATH=$blis_path/lib:$LD_LIBRARY_PATH

# Fixed parameters
size=5000
node=EPYC
export OMP_PLACES=cores

# Loop for both allocations: close and spread
for alloc in close spread
do
    export OMP_PROC_BIND=$alloc

    # Output path
    result_path=./fixed_matrix/$node
    if [ "$alloc" = "close" ]; then
        result_path="$result_path/close_cores"
    else
        result_path="$result_path/spread_cores"
    fi

    mkdir -p $result_path
    cd $result_path
    echo "============================================"
    echo "Running BLIS with OMP_PROC_BIND=$alloc"
    echo "Results will be stored in $result_path"
    echo "============================================"

    # Clean old CSVs
    rm -f blis_d.csv blis_f.csv

    # Run BLIS (float64 and float32)
    for idx in $(seq 1 1 128); do
        export OMP_NUM_THREADS=$idx
        echo "Running BLIS with $idx threads ($alloc)..."

        for iteration in $(seq 1 1 5); do
            echo -n "$idx," >> blis_d.csv
            $sbatch_path/../results/gemm_blis_d.exe $size $size $size

            echo -n "$idx," >> blis_f.csv
            $sbatch_path/../results/gemm_blis_f.exe $size $size $size

            echo "Iteration $iteration completed with $idx threads ($alloc)."
        done
    done

    echo "Finished BLIS runs for $alloc allocation."
    cd $sbatch_path
done

echo "BLIS Fixed Matrix job finished successfully for both close and spread."
