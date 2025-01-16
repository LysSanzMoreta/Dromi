#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils_omp.h"
#include <omp.h>



int main() {



    // Test the hash table ####################################

    // Create a hash table
    HashTable table;
    initHashTable(&table);
    hashAminoAcids(&table);


    // Load a file and pad the sequences
    char **seqs;
    int numSeqs = 0;
    int maxLength = 10;  // Desired length for each string

    // Read the file and store strings in the array
    readFileAndStoreStrings("seqs.txt", &seqs, &numSeqs, maxLength);
    padStringsToMaxLength(seqs, numSeqs, maxLength);


    // Retrieve values
    // printf("apple: %d\n", getHash(&table, "T"));    // Output: 10
    // printf("banana: %d\n", getHash(&table, "L"));  // Output: 20
    // printf("cherry: %d\n", getHash(&table, "#"));  // Output: 30
    // printf("orange: %d\n", getHash(&table, "W"));  // Output: -1 (not found)



    // run cosine_sim ###################################
    // double sim = 0.0;

    int N = 2000;

    // Allocate the array
    double* arrayA = (double*)malloc(N * sizeof(double));
    double* arrayB = (double*)malloc(N * sizeof(double));

    // Init a cosine result matrix
    // TODO: only init the upper triangle
    double **matrix = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        {
            matrix[i] = (double *)malloc(N * sizeof(double));
        }


    // Init test arrays to compute the cos sim
    // This is usually given with the
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        arrayA[i] = i;
        arrayB[i] = i;
    }
    // arrayB[N-1] = 0.0;


    // init matrix data
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }

    // printf("m %f\n", matrix[1][1]);

    double sim = cosine_sim(arrayA, arrayB, N);
    // printf("Sim: %f\n", sim);

    double val = 0.0;

    double time = omp_get_wtime();

    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < N; ++i){
    //     for (int j = 0; j < N; ++j){
    //         val = cosine_sim(arrayA, arrayB, N);
    //         matrix[i][j] = val;
    //         // printf("(%d, %d) sim: %f\n", i, j, val);
    //     }
    // }

    cosine_sim_3d(arrayA, arrayB, matrix, N, N);

    time = omp_get_wtime() - time;
    printf("%f\n", time);
    // printf("m %f\n", matrix[1][1]);




    // Free memory
    freeHashTable(&table);
    freeStrings(seqs, numSeqs);

    free(arrayA);
    free(arrayB);

    for (int i = 0; i < N; ++i){
        free(matrix[i]);
    }

    return 0;


    // Cosine example
        // // init an array

    // int N = 100;

    // // Allocate the array
    // double* arrayA = (double*)malloc(N * sizeof(double));
    // double* arrayB = (double*)malloc(N * sizeof(double));

    // #pragma omp parallel for
    // for (int i = 0; i < N; ++i){
    //     arrayA[i] = i;
    //     arrayB[i] = i;
    // }

    // // Compute the norm
    // double norm = 0.0;
    // norm2(arrayA, N, &norm);

    // printf("The norm is %f\n", norm);

    // double sim = 0.0;
    // cosine_sim(arrayA, arrayB, N, &sim);
    // printf("The cosine similarity is %f\n", sim);



}