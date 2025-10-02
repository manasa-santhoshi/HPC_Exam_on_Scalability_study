#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="fixedmatrix_thin"
#SBATCH --partition=THIN
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output="THIN_fixedmatrix_summary.out"

echo "Loading modules..."
module load openBLAS/0.3.26-omp
echo "Modules loaded."

sbatch_path=$(pwd)
blis_path=/u/dssc/msmaddi/blis_install
export LD_LIBRARY_PATH=$blis_path/lib:$LD_LIBRARY_PATH

# Fixed matrix size
size=5000
node=THIN
alloc=close   # change to "spread" if needed

export OMP_PLACES=cores
export OMP_PROC_BIND=$alloc

result_path=./fixed_matrix/$node
if [ "$alloc" = "close" ]; then
    result_path="$result_path/close_cores"
elif [ "$alloc" = "spread" ]; then
    result_path="$result_path/spread_cores"
fi

mkdir -p $result_path
cd $result_path
echo "Results will be stored in $result_path"

# Run experiments
for idx in $(seq 1 1 24); do
    export OMP_NUM_THREADS=$idx
    echo "Running with $idx cores..."

    for iteration in $(seq 1 1 5); do
        echo -n "$idx," >> openblas_d.csv
        $sbatch_path/../results/gemm_openblas_d.exe $size $size $size

        echo -n "$idx," >> openblas_f.csv
        $sbatch_path/../results/gemm_openblas_f.exe $size $size $size

        echo -n "$idx," >> blis_d.csv
        $sbatch_path/../results/gemm_blis_d.exe $size $size $size

        echo -n "$idx," >> blis_f.csv
        $sbatch_path/../results/gemm_blis_f.exe $size $size $size

        echo "Iteration $iteration completed with $idx cores."
    done
done

module purge
echo "THIN fixed_matrix job finished."
