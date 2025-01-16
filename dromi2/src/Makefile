TARGET_OMP	= dromi_omp
TARGET_CUDA = dromi_cuda


OBJS_OMP = dromi_omp.o utils_omp.o
OBJS_CUDA = dromi_cuda.o utils_cuda.o

OPT	= -g -O3
ISA	=
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) #$(XOPTS)

F90C  	= gfortran

CUDA_PATH ?= /appl/cuda/12.2.2
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common


all: $(TARGET_OMP) $(TARGET_CUDA)

$(TARGET_OMP): $(OBJS_OMP)
	$(CC) $(CFLAGS) -o $@ $(OBJS_OMP) -lm
$(TARGET_CUDA): $(OBJS_CUDA)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA) -lm

.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET_OMP)
	@rm -f $(TARGET_CUDA)


# dependencies
dromi_omp.o: dromi_omp.c utils_omp.h
utils_omp.o: utils_omp.c utils_omp.h
dromi_cuda.o: dromi_cuda.c utils_cuda.h
utils_cuda.o: utils_cuda.c utils_cuda.h