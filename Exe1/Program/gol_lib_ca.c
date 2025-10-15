#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "gol.h"

// Utility function to write PGM image
void write_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        exit(1);
    }
    fprintf(fp, "P5\n%d %d\n%d\n", xsize, ysize, maxval);
    unsigned char* buffer = (unsigned char*)malloc(xsize * ysize);
    for (int i = 0; i < xsize * ysize; i++) {
        buffer[i] = image[i] ? 0 : maxval;
    }
    fwrite(buffer, 1, xsize * ysize, fp);
    free(buffer);
    fclose(fp);
}

// Read PGM header
void read_header(int* xsize, int* ysize, int* maxval, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        exit(1);
    }
    char magic[3];
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Not a P5 PGM file\n");
        exit(1);
    }
    // Skip comments
    int c = fgetc(fp);
    while (c == '#' || c == '\n') {
        if (c == '#') {
            while (fgetc(fp) != '\n');
        }
        c = fgetc(fp);
    }
    ungetc(c, fp);
    fscanf(fp, "%d %d %d", xsize, ysize, maxval);
    fclose(fp);
}

// Read PGM image
void read_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }
    char magic[3];
    fscanf(fp, "%2s", magic);
    int x, y, m;
    int c = fgetc(fp);
    while (c == '#' || c == '\n') {
        if (c == '#') {
            while (fgetc(fp) != '\n');
        }
        c = fgetc(fp);
    }
    ungetc(c, fp);
    fscanf(fp, "%d %d %d", &x, &y, &m);
    fgetc(fp); // consume newline

    unsigned char* buffer = (unsigned char*)malloc(xsize * ysize);
    fread(buffer, 1, xsize * ysize, fp);

    for (int i = 0; i < xsize * ysize; i++) {
        image[i] = (buffer[i] < maxval / 2);
    }
    free(buffer);
    fclose(fp);
}

// Gather partial images from all processes (domain decomposition)
void gather_images(bool* partial, bool* full, int x, int y, int n_procs, int rank) {
    int rows_per_proc = y / n_procs;
    int remainder = y % n_procs;

    int* recvcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        recvcounts = (int*)malloc(n_procs * sizeof(int));
        displs = (int*)malloc(n_procs * sizeof(int));

        int offset = 0;
        for (int i = 0; i < n_procs; i++) {
            int local_rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = local_rows * x;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int sendcount = local_rows * x;

    MPI_Gatherv(partial, sendcount, MPI_C_BOOL, full, recvcounts, displs, 
                MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Initialize playground with random data (parallelized with OpenMP)
void initialise_playground(int x, int y, int maxval, const char* fname, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    double start_time = MPI_Wtime();

    // Domain decomposition: divide rows among processes
    int rows_per_proc = y / n_procs;
    int remainder = y % n_procs;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    bool* local_grid = (bool*)malloc(local_rows * x * sizeof(bool));

    // Parallelize initialization with OpenMP
    unsigned int seed = 12345 + rank * 7919;
    #pragma omp parallel
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < local_rows * x; i++) {
            local_grid[i] = (rand_r(&local_seed) % 100) < 50;
        }
    }

    // Gather all data to rank 0
    bool* full_grid = NULL;
    if (rank == 0) {
        full_grid = (bool*)malloc(x * y * sizeof(bool));
    }

    gather_images(local_grid, full_grid, x, y, n_procs, rank);

    // MPI Barrier to ensure all processes are done
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        write_pgm_image(full_grid, x, y, maxval, fname);
        free(full_grid);
        double end_time = MPI_Wtime();
        printf("Initialization completed in %.4f seconds\n", end_time - start_time);
    }

    free(local_grid);
    MPI_Finalize();
}

// Count alive neighbors with periodic boundary conditions
static inline int count_neighbors(bool* grid, int x, int y, int i, int j) {
    int count = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = (i + di + y) % y;
            int nj = (j + dj + x) % x;
            if (grid[ni * x + nj]) count++;
        }
    }
    return count;
}

// Static evolution with OpenMP and MPI parallelization
void static_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int x, y, maxval;
    read_header(&x, &y, &maxval, fname);

    // Domain decomposition
    int rows_per_proc = y / n_procs;
    int remainder = y % n_procs;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Allocate local grid with ghost rows (halo regions)
    bool* local_grid = (bool*)calloc((local_rows + 2) * x, sizeof(bool));
    bool* local_next = (bool*)calloc((local_rows + 2) * x, sizeof(bool));

    // Read initial state
    bool* full_grid = NULL;
    if (rank == 0) {
        full_grid = (bool*)malloc(x * y * sizeof(bool));
        read_pgm_image(full_grid, x, y, maxval, fname);
    }

    // Scatter data to all processes
    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(n_procs * sizeof(int));
        displs = (int*)malloc(n_procs * sizeof(int));
        int offset = 0;
        for (int i = 0; i < n_procs; i++) {
            int lr = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = lr * x;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(full_grid, sendcounts, displs, MPI_C_BOOL,
                 &local_grid[x], local_rows * x, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
        free(full_grid);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Determine neighbors for halo exchange
    int upper_neighbor = (rank - 1 + n_procs) % n_procs;
    int lower_neighbor = (rank + 1) % n_procs;

    // Evolution loop
    for (int step = 0; step < n_steps; step++) {
        // Exchange ghost rows (halo exchange)
        MPI_Sendrecv(&local_grid[x], x, MPI_C_BOOL, upper_neighbor, 0,
                     &local_grid[(local_rows + 1) * x], x, MPI_C_BOOL, lower_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&local_grid[local_rows * x], x, MPI_C_BOOL, lower_neighbor, 1,
                     local_grid, x, MPI_C_BOOL, upper_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Parallel evolution with OpenMP (static scheduling)
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < x; j++) {
                int neighbors = 0;
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = (j + dj + x) % x;
                        if (local_grid[ni * x + nj]) neighbors++;
                    }
                }

                bool current = local_grid[i * x + j];
                if (current) {
                    local_next[i * x + j] = (neighbors == 2 || neighbors == 3);
                } else {
                    local_next[i * x + j] = (neighbors == 3);
                }
            }
        }

        // Swap grids
        bool* temp = local_grid;
        local_grid = local_next;
        local_next = temp;

        // MPI Barrier to synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);

        // Save snapshot
        if ((step + 1) % snap_freq == 0) {
            bool* snapshot = NULL;
            if (rank == 0) {
                snapshot = (bool*)malloc(x * y * sizeof(bool));
            }
            gather_images(&local_grid[x], snapshot, x, y, n_procs, rank);

            if (rank == 0) {
                char snapshot_name[256];
                snprintf(snapshot_name, sizeof(snapshot_name), "snapshot_%05d.pgm", step + 1);
                write_pgm_image(snapshot, x, y, maxval, snapshot_name);
                free(snapshot);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Static evolution completed in %.4f seconds\n", end_time - start_time);
        printf("Steps: %d, Processes: %d, Threads: %d\n", n_steps, n_procs, omp_get_max_threads());
    }

    free(local_grid);
    free(local_next);
    MPI_Finalize();
}

// Ordered evolution (for comparison, less optimized)
void ordered_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int x, y, maxval;
    read_header(&x, &y, &maxval, fname);

    bool* grid = NULL;
    bool* next_grid = NULL;

    if (rank == 0) {
        grid = (bool*)malloc(x * y * sizeof(bool));
        next_grid = (bool*)malloc(x * y * sizeof(bool));
        read_pgm_image(grid, x, y, maxval, fname);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int step = 0; step < n_steps; step++) {
            // Sequential evolution on rank 0
            for (int i = 0; i < y; i++) {
                for (int j = 0; j < x; j++) {
                    int neighbors = count_neighbors(grid, x, y, i, j);
                    bool current = grid[i * x + j];
                    if (current) {
                        next_grid[i * x + j] = (neighbors == 2 || neighbors == 3);
                    } else {
                        next_grid[i * x + j] = (neighbors == 3);
                    }
                }
            }

            bool* temp = grid;
            grid = next_grid;
            next_grid = temp;

            if ((step + 1) % snap_freq == 0) {
                char snapshot_name[256];
                snprintf(snapshot_name, sizeof(snapshot_name), "snapshot_%05d.pgm", step + 1);
                write_pgm_image(grid, x, y, maxval, snapshot_name);
            }
        }

        double end_time = MPI_Wtime();
        printf("Ordered evolution completed in %.4f seconds\n", end_time - start_time);

        free(grid);
        free(next_grid);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
