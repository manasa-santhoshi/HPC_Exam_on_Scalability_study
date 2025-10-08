#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gol.h"

// --- I/O ---

void write_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("write_pgm_image"); exit(1); }
    fprintf(f, "P5\n# Game of Life\n%d %d\n%d\n", xsize, ysize, maxval);
    for (int i = 0; i < xsize * ysize; i++)
        fputc(image[i] ? 255 : 0, f);
    fclose(f);
}

void read_header(int* x, int* y, int* maxval, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("read_header"); exit(1); }
    char line[128];
    fgets(line, sizeof(line), f); // P5
    do { fgets(line, sizeof(line), f); } while (line[0] == '#');
    sscanf(line, "%d %d", x, y);
    fgets(line, sizeof(line), f); // maxval
    sscanf(line, "%d", maxval);
    fclose(f);
}

void read_pgm_image(bool* image, int x, int y, int maxval, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("read_pgm_image"); exit(1); }
    char line[128];
    fgets(line, sizeof(line), f); // P5
    do { fgets(line, sizeof(line), f); } while (line[0] == '#');
    fgets(line, sizeof(line), f); // maxval
    fread(image, sizeof(unsigned char), x * y, f);
    for (int i = 0; i < x * y; i++)
        image[i] = (image[i] > 0);
    fclose(f);
}

// --- MPI Gather Helper ---

void gather_images(bool* partial, bool* full, int x, int y, int n_procs, int rank) {
    int* counts = malloc(n_procs * sizeof(int));
    int* disps = malloc(n_procs * sizeof(int));
    int total = 0;
    for (int r = 0; r < n_procs; r++) {
        int rows = y / n_procs + (r < y % n_procs ? 1 : 0);
        counts[r] = rows * x;
        disps[r] = total;
        total += rows * x;
    }
    MPI_Gatherv(partial, counts[rank], MPI_C_BOOL,
                full, counts, disps, MPI_C_BOOL,
                0, MPI_COMM_WORLD);
    free(counts);
    free(disps);
}

// --- Initialization ---

void initialise_playground(int x, int y, int maxval, const char* fname, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &(int){0});
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int rows = y / n_procs + (rank < y % n_procs ? 1 : 0);
    bool* local = calloc(rows * x, sizeof(bool));
    srand(time(NULL) + rank);

#ifdef _OPENMP
    printf("[Rank %d] OpenMP threads = %d\n", rank, omp_get_max_threads());
#endif

    #pragma omp parallel for
    for (int i = 0; i < rows * x; i++)
        local[i] = (rand() % 2);

    if (rank == 0) {
        bool* full = malloc(x * y * sizeof(bool));
        gather_images(local, full, x, y, n_procs, rank);
        write_pgm_image(full, x, y, maxval, fname);
        free(full);
    } else {
        gather_images(local, NULL, x, y, n_procs, rank);
    }

    free(local);
    MPI_Finalize();
}

// --- Utility: Count Neighbors ---

int count_neighbors(const bool* grid, int row, int col, int x, int y) {
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (col + dx + x) % x;
            int ny = (row + dy + y) % y;
            sum += grid[ny * x + nx];
        }
    }
    return sum;
}

// --- Static Evolution (OpenMP + MPI) ---

void static_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &(int){0});
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int x, y, maxval;
    read_header(&x, &y, &maxval, fname);
    bool* global = malloc(x * y * sizeof(bool));
    read_pgm_image(global, x, y, maxval, fname);

    int rows = y / n_procs + (rank < y % n_procs ? 1 : 0);
    int start_row = rank * (y / n_procs) + (rank < y % n_procs ? rank : y % n_procs);
    bool* local = malloc((rows + 2) * x * sizeof(bool));

    for (int j = 0; j < rows; j++)
        memcpy(&local[(j + 1) * x], &global[(start_row + j) * x], x * sizeof(bool));

    memcpy(local, &global[((start_row - 1 + y) % y) * x], x * sizeof(bool));
    memcpy(&local[(rows + 1) * x], &global[((start_row + rows) % y) * x], x * sizeof(bool));

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    if (rank == 0)
        printf("Running static evolution with %d threads (rank %d)\n", n_threads, rank);
#endif

    double start_time = 0.0, elapsed = 0.0;

#ifdef _OPENMP
    start_time = omp_get_wtime();
#endif

    for (int step = 0; step < n_steps; step++) {
        bool* next = calloc(rows * x, sizeof(bool));

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < x; i++) {
                int neighbors = count_neighbors(local, j + 1, i, x, rows + 2);
                next[j * x + i] =
                    (neighbors == 3) ||
                    (local[(j + 1) * x + i] && neighbors == 2);
            }
        }

        memcpy(&local[x], next, rows * x * sizeof(bool));
        free(next);
    }

#ifdef _OPENMP
    elapsed = omp_get_wtime() - start_time;
#endif

    if (rank == 0) {
        FILE* f = fopen("times.csv", "a");
        if (f) {
            fprintf(f, "%d,%d,%d,%d,%lf\n", x, n_steps, n_procs,
#ifdef _OPENMP
                    n_threads,
#else
                    1,
#endif
                    elapsed);
            fclose(f);
        }
        printf("Elapsed time: %.3f seconds\n", elapsed);
    }

    free(local);
    free(global);
    MPI_Finalize();
}

// --- Ordered Evolution (for MPI correctness check) ---

void ordered_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
    static_evolution(fname, n_steps, snap_freq, argc, argv);
}
