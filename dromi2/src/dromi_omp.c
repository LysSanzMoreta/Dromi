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

    // Free memory
    freeHashTable(&table);
    freeStrings(seqs, numSeqs);

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