### BLIS Library
BLISROOT=${HOME}/blis_install
BLIS = -L$(BLISROOT)/lib -lpthread -lblis -lm

### Compiler flags
CFLAG = -m64 -fopenmp -DSAVE
INCLUDE = -I$(BLISROOT)/include/blis

FLOAT_FLAGS = gcc -DUSE_FLOAT
DOUBLE_FLAGS = gcc -DUSE_DOUBLE

### Source and Output
GEMM = gemm.c
DATA_FOLDER= ../results

BLIS_f = $(DATA_FOLDER)/gemm_blis_f.exe
BLIS_d = $(DATA_FOLDER)/gemm_blis_d.exe

.PHONY: float double clean

#--------------- FLOAT --------------------
float: $(BLIS_f)

$(BLIS_f): $(GEMM)
	$(FLOAT_FLAGS) -DBLIS $^ $(INCLUDE) $(BLIS) -o $@  $(CFLAGS)

#--------------- DOUBLE --------------------
double: $(BLIS_d)

$(BLIS_d): $(GEMM)
	$(DOUBLE_FLAGS) -DBLIS $^ $(INCLUDE) $(BLIS) -o $@  $(CFLAGS)


#--------------- CLEAN --------------------
clean:
	rm -f $(BLIS_f) $(BLIS_d)
