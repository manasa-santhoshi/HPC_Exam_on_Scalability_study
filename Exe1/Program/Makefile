# Makefile for Game of Life - HPC Exercise 1
# Place this in Program/ directory

CC = mpicc
CFLAGS = -O3 -fopenmp -Wall -std=c99
LDFLAGS = -fopenmp -lm

TARGET = gol.exe
SOURCES = game_of_life.c gol_lib.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = gol.h

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) *.pgm snapshot_*.pgm
	@echo "Cleaned build artifacts"

# Copy executable to test directories
#install: $(TARGET)
#	cp $(TARGET) ../OpenMP_scalability/THIN/
#	cp $(TARGET) ../OpenMP_scalability/EPYC/
#	cp $(TARGET) ../Strong_scalability/THIN/
#	cp $(TARGET) ../Strong_scalability/EPYC/
#	cp $(TARGET) ../Weak_scalability/THIN/
#	cp $(TARGET) ../Weak_scalability/EPYC/
#	@echo "Executable installed to all test directories"

# Quick test
test_init:
	mpirun -np 4 ./$(TARGET) -i -x 1000 -y 1000 -f test_init.pgm

test_run:
	mpirun -np 4 ./$(TARGET) -r -e 1 -f test_init.pgm -n 100 -s 20

.PHONY: all clean install test_init test_run
