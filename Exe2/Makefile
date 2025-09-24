CC = gcc
CFLAGS = -O3 -fopenmp
LIBS = -lopenblas -lm

# Results folder
OUTDIR = ../results

# Double precision
double:
	$(CC) $(CFLAGS) -DUSE_DOUBLE \
		-DFILENAME=\"$(OUTDIR)/openblas_d.csv\" \
		gemm.c -o gemm_openblas_d $(LIBS)

# Single precision
float:
	$(CC) $(CFLAGS) -DUSE_FLOAT \
		-DFILENAME=\"$(OUTDIR)/openblas_f.csv\" \
		gemm.c -o gemm_openblas_f $(LIBS)

# Clean binaries
clean:
	rm -f gemm_openblas_d gemm_openblas_f


