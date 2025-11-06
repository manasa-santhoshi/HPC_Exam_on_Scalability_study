# HPC Scalability Study

## Project Summary

This project focuses on analyzing the scalability of parallel programs on the **ORFEO HPC cluster**.  
Two main exercises were conducted to study performance and scalability across **EPYC** and **THIN** architectures.



### Exercise 1 – Game of Life

In this exercise, the *Game of Life* program was implemented using:

- **OpenMP** (shared-memory parallelism)  
- **MPI** (distributed-memory parallelism)  
- **Hybrid MPI + OpenMP**

The goal was to evaluate:

- **Strong scalability** – fixing the total problem size while increasing the number of threads/processes.  
- **Weak scalability** – increasing the problem size proportionally with the number of threads/processes.  

Performance metrics such as execution time, speedup, and efficiency were analyzed on both **EPYC** and **THIN** nodes.



### Exercise 2 – Matrix Multiplication (GEMM)

This part compares the performance of **BLIS** and **OpenBLAS** libraries for double and single precision matrix multiplication.

- Tested under both **fixed matrix size** and **fixed number of cores** conditions.  
- Executed on both **EPYC** and **THIN** architectures to analyze performance differences.



### Analysis

All performance data were collected as `.csv` files and visualized using Python scripts.  
Plots and analysis outputs are available in:

- `Exe1/Analysis/`  
- `Exe2/Analysis/`



## Repository Structure

HPC/
├── Exe1/
│ ├── OpenMP_scalability/
│ ├── Strong_scalability/
│ ├── Weak_scalability/
│ ├── scr/ # Source codes and SLURM scripts
│ └── Analysis/ # Plots and data analysis
│
├── Exe2/
│ ├── BLIS/ # BLIS performance results
│ ├── OpenBLAS/ # OpenBLAS performance results
│ └── Analysis/ # Comparative plots
│
├── Executables/
│ └── Compiled binaries
│
├── Project Report.pdf
└── README.md
