#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>



__global__ void cosine_sim(
const double* arrayA,
    const double* arrayB,
    double** matrix,
    int length_matrix,
    int length_features,
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;


}