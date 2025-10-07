#ifndef GOL_H
#define GOL_H

#include <stdbool.h>
#include <mpi.h>

void write_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename);
void read_header(int* xsize, int* ysize, int* maxval, const char* filename);
void read_pgm_image(bool* image, int xsize, int ysize, int maxval, const char* filename);
void gather_images(bool* partial, bool* full, int x, int y, int n_procs, int rank);
void initialise_playground(int x, int y, int maxval, const char* fname, int argc, char** argv);
void static_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv);
void ordered_evolution(const char* fname, int n_steps, int snap_freq, int argc, char** argv);

#endif
