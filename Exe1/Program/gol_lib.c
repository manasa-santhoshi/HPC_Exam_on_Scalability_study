#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>
#include <unistd.h>
#include "gol.h"

// --- I/O ---
void write_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("write_pgm_image"); exit(1); }
    fprintf(f, "P5\n# Game of Life\n%d %d %d\n", xsize, ysize, maxval);
    for (int i = 0; i < xsize * ysize; i++) {
        fputc(image[i] ? 255 : 0, f);
    }
    fclose(f);
}

void read_header(int* x, int* y, int* maxval, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("read_header"); exit(1); }
    char magic[3];
    fread(magic, 1, 2, f);
    if (magic[0] != 'P' || magic[1] != '5') { fprintf(stderr, "Not PGM P5\n"); exit(1); }
    char line[100];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d %d", x, y, maxval) == 3) break;
    }
    fclose(f);
}

void read_pgm_image(bool* image, int x, int y, int maxval, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("read_pgm_image"); exit(1); }
    char magic[3]; fread(magic, 1, 2, f);
    char line[100];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%*d %*d %*d") == 0) break;
    }
    int total = x * y;
    unsigned char* buf = malloc(total);
    fread(buf, 1, total, f);
    for (int i = 0; i < total; i++) image[i] = (buf[i] > 0);
    free(buf);
    fclose(f);
}

// --- MPI ---
void gather_images(bool* partial, bool* full, int x, int y, int n_procs, int rank) {
    if (rank == 0) {
        int* counts = calloc(n_procs, sizeof(int));
        int* disps = calloc(n_procs, sizeof(int));
        for (int r = 0; r < n_procs; r++) {
            int rows = y / n_procs + (r < y % n_procs ? 1 : 0);
            counts[r] = rows * x;
            disps[r] = r * (y / n_procs) * x + (r < y % n_procs ? r : y % n_procs) * x;
        }
        MPI_Gatherv(partial, counts[rank], MPI_C_BOOL, full, counts, disps, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        free(counts); free(disps);
    } else {
        int rows = y / n_procs + (rank < y % n_procs ? 1 : 0);
        MPI_Gatherv(partial, rows * x, MPI_C_BOOL, NULL, NULL, NULL, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }
}

// --- Initialisation ---
void initialise_playground(int x, int y, int maxval, const char* fname, int argc, char** argv) {
    int rank, n_procs;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &(int){0});
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int rows = y / n_procs + (rank < y % n_procs ? 1 : 0);
    bool* local = calloc(rows * x, sizeof(bool));
    srand(time(NULL) + rank);
    #pragma omp parallel for
    for (int i = 0; i < rows * x; i++) local[i] = (rand() % 2 == 0);

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

// --- Static Evolution ---
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

void static_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
    int rank, n_procs, n_threads = 1;
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

    // Halo init
    for (int j = 0; j < rows; j++)
        memcpy(&local[(j+1)*x], &global[(start_row+j)*x], x * sizeof(bool));
    memcpy(local, &global[((start_row-1+y)%y)*x], x * sizeof(bool)); // top halo
    memcpy(&local[(rows+1)*x], &global[((start_row+rows)%y)*x], x * sizeof(bool)); // bottom halo

    double start_time = 0.0;
    #ifdef TIME
    if (rank == 0) start_time = omp_get_wtime();
    #endif

    for (int step = 0; step < n_steps; step++) {
        bool* next = calloc(rows * x, sizeof(bool));
        #pragma omp parallel for
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < x; i++) {
                int neighbors = count_neighbors(local, j+1, i, x, rows+2);
                next[j*x + i] = (neighbors == 2 || neighbors == 3);
            }
        }

        // Halo exchange
        bool* top_halo = malloc(x * sizeof(bool));
        bool* bot_halo = malloc(x * sizeof(bool));
        memcpy(top_halo, next, x * sizeof(bool));
        memcpy(bot_halo, &next[(rows-1)*x], x * sizeof(bool));

        int prev = (rank - 1 + n_procs) % n_procs;
        int next_rank = (rank + 1) % n_procs;
        MPI_Sendrecv(bot_halo, x, MPI_C_BOOL, next_rank, 0,
                     &local[(rows+1)*x], x, MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(top_halo, x, MPI_C_BOOL, prev, 0,
                     local, x, MPI_C_BOOL, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        free(top_halo); free(bot_halo);
        memcpy(&local[x], next, rows * x * sizeof(bool));
        free(next);
    }

    #ifdef TIME
    if (rank == 0) {
        double elapsed = omp_get_wtime() - start_time;
        FILE* f = fopen("times.csv", "a");
        fprintf(f, "1,%d,%d,%d,%d,%lf\n", x, n_steps, n_procs, n_threads, elapsed);
        fclose(f);
    }
    #endif

    free(local); free(global);
    MPI_Finalize();
}

// --- Ordered Evolution (serial logic, MPI-only) ---
void ordered_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv) {
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

    // Initialize local with halo rows
    for (int j = 0; j < rows; j++)
        memcpy(&local[(j+1)*x], &global[(start_row+j)*x], x * sizeof(bool));
    memcpy(local, &global[((start_row - 1 + y) % y)*x], x * sizeof(bool)); // top halo
    memcpy(&local[(rows+1)*x], &global[((start_row + rows) % y)*x], x * sizeof(bool)); // bottom halo

    double start_time = 0.0;
    #ifdef TIME
    if (rank == 0) start_time = MPI_Wtime();
    #endif

    for (int step = 0; step < n_steps; step++) {
        // Cycle over processes in order: 0, 1, ..., n_procs-1
        for (int turn = 0; turn < n_procs; turn++) {
            if (rank == turn) {
                // Evolve local rows sequentially (row-major)
                for (int j = 1; j <= rows; j++) {
                    for (int i = 0; i < x; i++) {
                        int neighbors = count_neighbors(local, j, i, x, rows + 2);
                        local[j * x + i] = (neighbors == 2 || neighbors == 3);
                    }
                }

                // Send bottom halo to next process (if exists)
                if (n_procs > 1) {
                    int next = (turn + 1) % n_procs;
                    int prev = (turn - 1 + n_procs) % n_procs;
                    bool* bot_halo = &local[(rows) * x]; // last evolved row
                    bool* top_halo = malloc(x * sizeof(bool));

                    if (turn == n_procs - 1) {
                        // Last process sends to rank 0
                        MPI_Send(bot_halo, x, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD);
                        MPI_Recv(top_halo, x, MPI_C_BOOL, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        memcpy(local, top_halo, x * sizeof(bool)); // update top halo
                    } else if (turn == 0) {
                        // First process receives from last
                        MPI_Recv(top_halo, x, MPI_C_BOOL, n_procs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(bot_halo, x, MPI_C_BOOL, next, 0, MPI_COMM_WORLD);
                        memcpy(local, top_halo, x * sizeof(bool));
                    } else {
                        // Middle processes
                        MPI_Send(bot_halo, x, MPI_C_BOOL, next, 0, MPI_COMM_WORLD);
                        MPI_Recv(top_halo, x, MPI_C_BOOL, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        memcpy(local, top_halo, x * sizeof(bool));
                    }
                    free(top_halo);
                } else {
                    // Single process: wrap-around
                    memcpy(local, &local[rows * x], x * sizeof(bool)); // top = last evolved
                    memcpy(&local[(rows+1)*x], &local[x], x * sizeof(bool)); // bottom = first evolved
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    #ifdef TIME
    if (rank == 0) {
        double elapsed = MPI_Wtime() - start_time;
        FILE* f = fopen("times_ordered.csv", "a");
        fprintf(f, "0,%d,%d,%d,1,%lf\n", x, n_steps, n_procs, elapsed);
        fclose(f);
    }
    #endif

    free(local); free(global);
    MPI_Finalize();
}
