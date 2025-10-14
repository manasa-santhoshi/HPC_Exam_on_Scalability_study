#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

// From provided utility
void write_pgm_image(void *image, int maxval, int xsize, int ysize, const char *image_name);
void read_pgm_image(void **image, int *maxval, int *xsize, int *ysize, const char *image_name);

// Constants
#define INIT 1
#define RUN  2
#define ORDERED 0
#define STATIC  1
#define MAX_FILENAME 256

// Global args (set by parse_args)
int action = 0;
int k = 100;          // grid size k x k
int e = STATIC;       // evolution type
int nsteps = 100;     // number of steps
int save_freq = 0;    // save every N steps (0 = only final)
char fname[MAX_FILENAME] = "game_of_life.pgm";

// Parse command line
void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "irk:e:f:n:s:")) != -1) {
        switch (c) {
            case 'i': action = INIT; break;
            case 'r': action = RUN; break;
            case 'k': k = atoi(optarg); break;
            case 'e': e = atoi(optarg); break;
            case 'f': strncpy(fname, optarg, MAX_FILENAME - 1); break;
            case 'n': nsteps = atoi(optarg); break;
            case 's': save_freq = atoi(optarg); break;
            default: fprintf(stderr, "Unknown option -%c\n", c); exit(1);
        }
    }
    if (action == 0) {
        fprintf(stderr, "Error: must specify -i or -r\n");
        exit(1);
    }
}

// Count live neighbors (with periodic BC in cols, halo in rows)
int count_neighbors(unsigned char *grid, int i, int j, int local_nrows, int n) {
    int count = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = i + di;
            int nj = (j + dj + n) % n; // periodic in columns

            // Handle row boundaries via halo
            if (ni < 0) ni = local_nrows;       // top halo
            else if (ni > local_nrows + 1) ni = 1; // bottom halo (should not happen)

            count += (grid[ni * n + nj] == 255) ? 1 : 0;
        }
    }
    return count;
}

// Exchange halo rows
void exchange_halos(unsigned char *grid, int local_nrows, int n, int rank, int size) {
    int top = (rank - 1 + size) % size;
    int bot = (rank + 1) % size;

    MPI_Sendrecv(
        &grid[(local_nrows) * n], n, MPI_UNSIGNED_CHAR, bot, 0,
        &grid[0],                n, MPI_UNSIGNED_CHAR, top, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        &grid[1 * n], n, MPI_UNSIGNED_CHAR, top, 1,
        &grid[(local_nrows + 1) * n], n, MPI_UNSIGNED_CHAR, bot, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// Save full grid (gather on rank 0)
void save_full_grid(unsigned char *local_grid, int local_nrows, int n, int rank, int size, const char *filename) {
    if (rank == 0) {
        unsigned char *full = malloc(n * n);
        memcpy(full, &local_grid[1 * n], local_nrows * n);

        int offset = local_nrows;
        for (int r = 1; r < size; r++) {
            int recv_rows = (r == size - 1) ? (n - (size - 1) * ((n + size - 1) / size)) : ((n + size - 1) / size);
            if (recv_rows <= 0) recv_rows = (n + size - 1) / size;
            MPI_Recv(&full[offset * n], recv_rows * n, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += recv_rows;
        }
        write_pgm_image(full, 255, n, n, filename);
        free(full);
    } else {
        MPI_Send(&local_grid[1 * n], local_nrows * n, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
}

// Save snapshot with step number
void save_snapshot(unsigned char *local_grid, int local_nrows, int n, int rank, int size, int step) {
    char filename[256];
    snprintf(filename, sizeof(filename), "snapshot_%05d.pgm", step);
    save_full_grid(local_grid, local_nrows, n, rank, size, filename);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Simple block row distribution
    int local_nrows = (k + size - 1) / size;
    if (rank == size - 1) {
        local_nrows = k - (size - 1) * local_nrows;
    }
    if (local_nrows <= 0) local_nrows = (k + size - 1) / size;

    unsigned char *current = calloc((local_nrows + 2) * k, sizeof(unsigned char));
    unsigned char *next    = calloc((local_nrows + 2) * k, sizeof(unsigned char));

    if (action == INIT) {
        // Initialize random grid (0/255)
        unsigned int seed = 12345 + rank;
        for (int i = 0; i < local_nrows; i++) {
            for (int j = 0; j < k; j++) {
                current[(i + 1) * k + j] = (rand_r(&seed) % 2) ? 255 : 0;
            }
        }
        save_full_grid(current, local_nrows, k, rank, size, fname);
        if (rank == 0) printf("✅ Initialized %s (%dx%d)\n", fname, k, k);
    }

    if (action == RUN) {
        // Read full grid on rank 0, then scatter
        if (rank == 0) {
            void *full_grid = NULL;
            int maxval, xsize, ysize;
            read_pgm_image(&full_grid, &maxval, &xsize, &ysize, fname);
            if (xsize != k || ysize != k) {
                fprintf(stderr, "Grid size mismatch: expected %dx%d, got %dx%d\n", k, k, xsize, ysize);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Send blocks to ranks
            int offset = 0;
            for (int r = 0; r < size; r++) {
                int send_rows = (r == size - 1) ? (k - (size - 1) * ((k + size - 1) / size)) : ((k + size - 1) / size);
                if (send_rows <= 0) send_rows = (k + size - 1) / size;
                if (r == 0) {
                    memcpy(&current[1 * k], (unsigned char*)full_grid + offset * k, send_rows * k);
                } else {
                    MPI_Send((unsigned char*)full_grid + offset * k, send_rows * k, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
                }
                offset += send_rows;
            }
            free(full_grid);
        } else {
            MPI_Recv(&current[1 * k], local_nrows * k, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Timing
        double start_time = MPI_Wtime();

        // Main evolution loop
        for (int step = 0; step < nsteps; step++) {
            exchange_halos(current, local_nrows, k, rank, size);

            // Static evolution (double buffer)
            #pragma omp parallel for
            for (int i = 1; i <= local_nrows; i++) {
                for (int j = 0; j < k; j++) {
                    int neighbors = count_neighbors(current, i, j, local_nrows, k);
                    if (current[i * k + j] == 255) {
                        next[i * k + j] = (neighbors == 2 || neighbors == 3) ? 255 : 0;
                    } else {
                        next[i * k + j] = (neighbors == 3) ? 255 : 0;
                    }
                }
            }

            // Swap buffers
            unsigned char *tmp = current;
            current = next;
            next = tmp;

            // Save snapshot
            if (save_freq > 0 && (step + 1) % save_freq == 0) {
                save_snapshot(current, local_nrows, k, rank, size, step + 1);
            }
        }

        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;

        // Save final state
        if (save_freq == 0) {
            save_snapshot(current, local_nrows, k, rank, size, nsteps);
        }

        // Log timing (for CSV)
        if (rank == 0) {
            printf("✅ Completed %d steps in %.4f seconds\n", nsteps, total_time);
            // Optional: write to CSV
            FILE *csv = fopen("results.csv", "a");
            if (csv) {
                fprintf(csv, "%d,%d,%d,%d,%d,%.6f\n", k, nsteps, size, omp_get_max_threads(), e, total_time);
                fclose(csv);
            }
        }
    }

    free(current);
    free(next);
    MPI_Finalize();
    return 0;
}
