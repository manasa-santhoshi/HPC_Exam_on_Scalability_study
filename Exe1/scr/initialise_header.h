#ifndef INITIALISE
#define INITIALISE

#include <mpi.h>

/**
 * Writes a PGM image in parallel using MPI I/O.
 *
 * Each MPI process writes its local portion of the image at the correct
 * offset so the final file is contiguous and correct, even if processes
 * have different row counts.
 */
void write_pgm_image_parallel(unsigned char* image,
                              int xwidth, int ywidth,
                              int rows_per_proc,
                              int rank, int n_procs,
                              int maxval,
                              const char* filename);

/**
 * Initializes the Game of Life playground in parallel using MPI + OpenMP.
 * - Creates a random grid of 0s and 1s
 * - Saves the grid as a PGM file using write_pgm_image_parallel()
 */
void initialise_playground(int k, int maxval, char* fname, int argc, char **argv);

#endif
