#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils_omp.h"
#include <omp.h>



int main() {
    // dimensions for tensors and matrices
    int B = 10; // num batches
    int N = 6; // num seqs per batch
    int M = 20; // max seq lenth
    int K = 5;  // feature length

    char* filenames[10];

    for (int i = 0; i < 10; i++) {
        // Allocate memory for each filename and create the pattern "fileX.bin"
        filenames[i] = malloc(13 * sizeof(char));
        sprintf(filenames[i], "data_%d.bin", i);
        printf("%s\n", filenames[i]);
    }


    // Init a result matrix
    float **matrix = (float **)malloc(N * sizeof(float *));
    for (int i = 0; i < N; i++)
        {
            matrix[i] = (float *)malloc(N * sizeof(float));
        }

    // init 2 tensors for the 3d calculations
    float ***tensorA = (float ***)malloc(N * sizeof(float **));
    float ***tensorB = (float ***)malloc(N * sizeof(float **));

    for (int i = 0; i < N; i++){
            tensorA[i] = (float **)malloc(M * sizeof(float *));
            tensorB[i] = (float **)malloc(M * sizeof(float *));
            for (int j = 0; j < M; j++){
                tensorA[i][j] = (float *)malloc(K * sizeof(float));
                tensorB[i][j] = (float *)malloc(K * sizeof(float));
            }
        }


    // init matrix data
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }

    // // init the tensor data
    // #pragma omp parallel for collapse(3)
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < M; j++)
    //     {
    //         for (int k = 0; k < K; k++)
    //         {
    //             tensorA[i][j][k] = 1.0;
    //             tensorB[i][j][k] = 1.0;
    //         }
    //     }
    // }
    // tensorB[0][19][0] = 0.0;
    // tensorB[0][18][0] = 0.0;
    // tensorB[0][17][0] = 0.0;


    // Run the batched cosine sim
    float time = omp_get_wtime();

    if (B > 2) {
        for (int i = 0; i < B; ++i)
        {
            load_tensor_binary(filenames[i], &tensorA, &N, &M, &K);
            for (int j = i+1; j < B; ++j)
            {
                // TODO: Load the data into the 2 tensors
                load_tensor_binary(filenames[j], &tensorB, &N, &M, &K);

                // Compute the similarities
                cosine_sim_3d_masked(tensorA, tensorB, matrix, N, M, K);
                printf("batch: %d vs %d\n", i, j);

                // TODO: save the matrix as a
                // binary file with the batch ids


            }
        }
    } else {
        // TODO: Load only 1 tensor since B=1


        // Compute the similarities
        cosine_sim_3d(tensorA, tensorA, matrix, N, M, K);


        // TODO: save the matrix as a
        // binary file with the batch ids

    }


    time = omp_get_wtime() - time;
    printf("time: %f\n", time);
    printf("tensor %f\n", tensorA[1][2][3]);
    printf("matrix %f\n", matrix[0][0]);
    printf("matrix %f\n", matrix[1][2]);




    // Free memory again

    for (int i = 0; i < N; ++i){
        free(matrix[i]);
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            free(tensorA[i][j]);
            free(tensorB[i][j]);
        }
    }

    return 0;
}