#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>



__global__ void cosine_sim_3d(
    const float* arrayA,
    const float* arrayB,
    float** matrix,
    int length_matrix,
    int length_features,
){
    extern __shared__ Complex shared_mem[]; // Use shared memory
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.y * blockIdx.y + threadIdx.y;







}