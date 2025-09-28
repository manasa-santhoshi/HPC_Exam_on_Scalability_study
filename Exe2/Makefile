### BLIS Library
BLISROOT=${HOME}/blis_install
BLIS = -L$(BLISROOT)/lib -lpthread -lblis -lm


### OpenBLAS Library
OPENBLASROOT=$(OPENBLAS_ROOT)
OPENBLAS = -L$(OPENBLASROOT)/lib -lpthread -lopenblas -lm
OPENBLASINC = -I$(OPENBLASROOT)/include

### Compiler flags
CFLAGS = -m64 -fopenmp -DSAVE
INCLUDE = -I$(BLISROOT)/include/blis

FLOAT_FLAGS = gcc -DUSE_FLOAT
DOUBLE_FLAGS = gcc -DUSE_DOUBLE

### Source and Output
GEMM = gemm.c
DATA_FOLDER= ../results

BLIS_f = $(DATA_FOLDER)/gemm_blis_f.exe
BLIS_d = $(DATA_FOLDER)/gemm_blis_d.exe
OPENBLAS_f = $(DATA_FOLDER)/gemm_openblas_f.exe
OPENBLAS_d = $(DATA_FOLDER)/gemm_openblas_d.exe

.PHONY: float double clean

#--------------- FLOAT --------------------
float: $(BLIS_f)

$(BLIS_f): $(GEMM)
	$(FLOAT_FLAGS) -DBLIS $^ $(INCLUDE) $(BLIS) -o $@  $(CFLAGS)

#--------------- DOUBLE --------------------
double: $(BLIS_d)

$(BLIS_d): $(GEMM)
	$(DOUBLE_FLAGS) -DBLIS $^ $(INCLUDE) $(BLIS) -o $@  $(CFLAGS)


#--------------- FLOAT (OpenBLAS) --------------------
openblas_float: $(OPENBLAS_f)

$(OPENBLAS_f): $(GEMM)
	$(FLOAT_FLAGS) -DOPENBLAS $^ $(OPENBLASINC) $(OPENBLAS) -o $@ $(CFLAGS)

#--------------- DOUBLE (OpenBLAS) --------------------
openblas_double: $(OPENBLAS_d)

$(OPENBLAS_d): $(GEMM)
	$(DOUBLE_FLAGS) -DOPENBLAS $^ $(OPENBLASINC) $(OPENBLAS) -o $@ $(CFLAGS)


#--------------- CLEAN --------------------
clean:
	rm -f $(BLIS_f) $(BLIS_d) $(OPENBLAS_f) $(OPENBLAS_d)
