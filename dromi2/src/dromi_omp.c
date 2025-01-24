#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils_omp.h"
#include <omp.h>


void print_help() {
    printf(
        "Usage: ./dromi_omp \
         <int: num batches> \
         <int: num seqs per batch> \
         <int: max seq length> \
         <feature length> \
         <char: base of batch filename> \
         <char: path>\n"
        );
    printf("or\n");
    printf("Usage: ./dromi_omp -h\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  <int: num batches>   number of batch files inside the folder\n");
    printf("  <int: num seqs per batch>   1st tensor dimension: the number of sequences per batch\n");
    printf("  <int: max seq length>   2nd tensor dimension: the length of each (padded) sequence\n");
    printf("  <feature length>   3rd tensor dimension: the number of features \
    which encode each character of the sequence\n");
    printf("  <char: base of batch filename> \
    All files must be 3d tensors with the defined dimensions \
    (num seqs per batch, max seq length, feature length) \
    you can use the python helper tool to convert your data to this format. \
    All files must be in the same folder <path> and adhere to the convention: \
    {base of batch filename}{i}.bin. See example below.\n");
    printf("  <char: path>   The folder/path where the batch files are located \n");
    printf("\n");
    printf("Example:\n");
    printf("  ./dromi_omp 10 6 20 5 data_ ../test_data/ \n");
    printf("This runs dromi with 10 batches of each 6 sequences, \
        maximum sequence length of 20 and a feature dimension of 5. \
        Thus, the tensor dimensions of one batch are (6, 20, 5) here. \
        The files are named data_0.bin ... data_9.bin and reside inside \
        the folder test_data/\n");
}


int main(int argc, char* argv[]) {
    // dimensions for tensors and matrices
    // int B = 10; // num batches
    // int N = 6; // num seqs per batch
    // int M = 20; // max seq lenth
    // int K = 5;  // feature length
    if (strcmp(argv[1], "-h") == 0) {
        // If -h is passed, print the help message
        print_help();
        return 0;
    }

    int B = atoi(argv[1]); // num batches
    int N = atoi(argv[2]); // num seqs per batch
    int M = atoi(argv[3]); // max seq lenth
    int K = atoi(argv[4]);  // feature length
    // the base of the batch filename.
    char* fileName = argv[5];
    char* path = argv[6];

    char* filenames[B];

    for (int i = 0; i < B; i++) {
        // Allocate memory for each filename and create the pattern "fileX.bin"
        filenames[i] = malloc(256 * sizeof(char));
        sprintf(filenames[i], "%s/%s%d.bin", path, fileName, i);
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

    // Run the batched cosine sim
    // float time = omp_get_wtime();

    if (B > 2) {
        for (int i = 0; i < B; ++i)
        {
            load_tensor_binary(filenames[i], &tensorA, &N, &M, &K);
            for (int j = i+1; j < B; ++j)
            {
                load_tensor_binary(filenames[j], &tensorB, &N, &M, &K);

                // Compute the similarities
                cosine_sim_3d_masked(tensorA, tensorB, matrix, N, M, K);

                // binary file with the batch ids
                char saveFilename[256];
                sprintf(saveFilename, "%s/similarities_%d_%d.bin", path, i, j);
                save_matrix_binary(
                    saveFilename,
                    &matrix,
                    &N,
                    &N
                );
            }
        }
    } else {
        // TODO: Load only 1 tensor since B=1
        load_tensor_binary(filenames[0], &tensorA, &N, &M, &K);

        // Compute the similarities
        cosine_sim_3d(tensorA, tensorA, matrix, N, M, K);

        // Save the file:
        char saveFilename[256];
        sprintf(saveFilename, "%s/similarities_0_0.bin", path);
        save_matrix_binary(
            saveFilename,
            &matrix,
            &N,
            &N
        );

    }


    // time = omp_get_wtime() - time;
    // printf("time needed: %f sec\n", time);




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

    for (int i = 0; i < B; i++) {
        free(filenames[i]);
    }

    return 0;
}