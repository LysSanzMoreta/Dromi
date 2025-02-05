#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "utils_omp.h"
#include <omp.h>


#define TABLE_SIZE 64

void norm2(const float* array, int length, float* val){
    float sumOfSquares = 0.0;
    #pragma omp parallel for reduction(+:sumOfSquares)
    for (int i = 0; i < length; ++i){
        sumOfSquares += array[i]*array[i];
    }
    *val = sqrt(sumOfSquares);
}


void printArray(const float* array, int N) {
    for (int i = 0; i < N; ++i) {
        printf("%f ", array[i]);
    }
    printf("\n");
}


float cosine_sim(float* arrayA, float* arrayB, int length){
    float dotProduct = 0.0, normA = 0.0, normB = 0.0;

    // #pragma omp parallel for reduction(+:dotProduct, normA, normB)
    for (int i = 0; i < length; i++) {
        dotProduct += arrayA[i] * arrayB[i];
        normA += arrayA[i] * arrayA[i];
        normB += arrayB[i] * arrayB[i];
    }
    // Compute the norms
    normA = sqrtf(normA);
    normB = sqrtf(normB);
    return dotProduct / (normA * normB);
}

void cosine_sim_2d(
    float** matrixA,
    float** matrixB,
    float** result_matrix,
    int num_arrays,
    int length_arrays
    ){

    // Only compute the upper triangle
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_arrays; ++i){
        for (int j = i+1; j < num_arrays; ++j){
            result_matrix[i][j] = cosine_sim(matrixA[i], matrixB[j], length_arrays);
        }}
}


void cosine_sim_3d(
    float*** tensorA,
    float*** tensorB,
    float** result_matrix,
    int num_arrays,
    int length_arrays,
    int length_features
    ){

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_arrays; ++i)
    {
        for (int j = 0; j < num_arrays; ++j)
        {
            // Now, we are working with a matrix of
            // dim: (length_arrays, length_arrays, length_features)
            // in the simple case, we just take the mean of the
            // cos_sims
            // TODO: only compute the diag over the length_arrays
            // we could also iter over a neighbourhood of vals
            float sum = 0.0;
            float normaliser = (float)length_arrays;
            normaliser = normaliser*normaliser;

            #pragma omp parallel for collapse(2)
            for (int k = 0; k < length_arrays; ++k)
            {
                for (int l = 0; l < length_arrays; ++l)
                {
                    sum += cosine_sim(tensorA[i][k], tensorB[j][l], length_features);
                }
            }
            result_matrix[i][j] = sum/normaliser;
        }
    }
}


void cosine_sim_3d_masked(
    float*** tensorA,
    float*** tensorB,
    float** result_matrix,
    int num_arrays,
    int length_arrays,
    int length_features
    ){

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_arrays; ++i)
    {
        for (int j = 0; j < num_arrays; ++j)
        {
            // Now, we are working with a matrix of
            // dim: (length_arrays, length_arrays, length_features)
            // in the simple case, we just take the mean of the
            // cos_sims
            // TODO: only compute the diag over the length_arrays
            // we could also iter over a neighbourhood of vals
            float sum = 0.0;
            int normaliser = length_arrays;

            #pragma omp parallel for collapse(2)
            for (int k = 0; k < length_arrays; ++k)
            {
                for (int l = 0; l < length_arrays; ++l)
                {
                    if(k==l)
                    {
                        if(tensorA[i][k][0] != 0.0 && tensorB[i][k][0] != 0.0)
                        {
                            sum += cosine_sim(tensorA[i][k], tensorB[j][l], length_features);
                        } else {
                            normaliser -= 1;
                        }
                    }
                }
            }
            result_matrix[i][j] = sum/(float)normaliser;
        }
    }
}



void load_tensor_binary(
    const char *filename,
    float ****tensor,
    int *x_dim,
    int *y_dim,
    int *z_dim
    ) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // Read the dimensions (3 integers)
    if (fread(x_dim, sizeof(int), 1, file) != 1 ||
        fread(y_dim, sizeof(int), 1, file) != 1 ||
        fread(z_dim, sizeof(int), 1, file) != 1) {
        perror("Error reading dimensions");
        exit(1);
    }

    // Read the tensor data from the file into the 3D tensor
    for (int i = 0; i < *x_dim; i++) {
        for (int j = 0; j < *y_dim; j++) {
            if (fread((*tensor)[i][j], sizeof(float), *z_dim, file) != *z_dim) {
                perror("Error reading tensor data");
                exit(1);
            }
        }
    }

    fclose(file);
}


void save_matrix_binary(
    const char *filename,
    float ***matrix,
    int *x_dim,
    int *y_dim
) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        exit(1);
    }

    // Write dimensions of the tensor
    fwrite(&x_dim, sizeof(int), 1, file);
    fwrite(&y_dim, sizeof(int), 1, file);

    // Write the data (assuming the matrix is a 1D array representing the 2D matrix)
    size_t matrix_size = *x_dim * *y_dim ;
    fwrite(matrix, sizeof(float), matrix_size, file);

    fclose(file);
}

// void cosine_sim_3d_batch(
//     float*** tensorA,
//     float*** tensorB,
//     float** result_matrix,
//     int batch_size,
//     int num_arrays,
//     int length_arrays,
//     int length_features
//     ){

//     // Only compute the upper triangle
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < num_arrays; ++i)
//     {
//         for (int j = 0; j < num_arrays; ++j)
//         {
//             // Now, we are working with a matrix of
//             // dim: (batch_size, length_arrays, length_arrays, length_features)
//             // in the simple case, we just take the mean of the
//             // cos_sims
//             // TODO: only compute the diag over the length_arrays
//             // we could also iter over a neighbourhood of vals
//             float sum = 0.0;
//             float normaliser = (float)length_arrays;
//             normaliser = normaliser*normaliser;
//             #pragma omp parallel for collapse(2)
//             for (int k = 0; k < length_arrays; ++k)
//             {
//                 for (int l = 0; l < length_arrays; ++l)
//                 {
//                     sum += cosine_sim(tensorA[i][k], tensorB[j][l], length_features);
//                 }
//             }
//             result_matrix[i][j] = sum/normaliser;
//         }
//     }
// }


void readFileAndStoreStrings(const char *filename, char ***strings, int *numStrings, int maxLength) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(1); // Exit if file can't be opened
    }

    char buffer[1024]; // Temporary buffer to read lines
    *numStrings = 0;
    *strings = malloc(10 * sizeof(char*)); // Start with space for 10 strings, can be reallocated

    while (fgets(buffer, sizeof(buffer), file)) {
        // Remove newline character from the end of the line
        buffer[strcspn(buffer, "\n")] = '\0';

        // Allocate memory for the string and copy the line
        (*strings)[*numStrings] = malloc((strlen(buffer) + 1) * sizeof(char)); // +1 for the null terminator
        strcpy((*strings)[*numStrings], buffer);

        // Resize the array to store more strings if needed
        (*numStrings)++;
        if (*numStrings % 10 == 0) { // Resize in blocks of 10 for efficiency
            *strings = realloc(*strings, (*numStrings + 10) * sizeof(char*));
        }
    }

    fclose(file);
}


void padStringsToMaxLength(char **strings, int numStrings, int maxLength) {
    #pragma omp parallel for
    for (int i = 0; i < numStrings; i++) {
        int currentLength = strlen(strings[i]);
        if (currentLength < maxLength) {
            // Allocate new space for the padded string (maxLength + 1 for null terminator)
            strings[i] = realloc(strings[i], maxLength + 1);

            // Add padding characters to the string
            for (int j = currentLength; j < maxLength; j++) {
                strings[i][j] = '#';
            }
            strings[i][maxLength] = '\0';  // Null-terminate the string
        }
    }
}


void freeStrings(char **strings, int numStrings) {
    for (int i = 0; i < numStrings; i++) {
        free(strings[i]); // Free each string
    }
    free(strings); // Free the array of strings
}


// Hash table functions
unsigned int hash(const char *key) {
    unsigned int hashValue = 0;
    while (*key) {
        hashValue = hashValue * 31 + (*key++);
    }
    return hashValue % TABLE_SIZE;
}

void initHashTable(HashTable *table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        table->items[i] = NULL;
    }
}


void insertHash(HashTable *table, const char *key, int value) {
    unsigned int index = hash(key);
    HashItem *newItem = (HashItem *)malloc(sizeof(HashItem));
    newItem->key = strdup(key);  // Copy the key string
    newItem->value = value;
    table->items[index] = newItem;
}


int getHash(HashTable *table, const char *key) {
    unsigned int index = hash(key);
    if (table->items[index] != NULL && strcmp(table->items[index]->key, key) == 0) {
        return table->items[index]->value;
    } else {
        return -1;  // Key not found, return -1
    }
}

void hashAminoAcids(HashTable *table){
    // Insert the amino acid mappings into the hash table
    insertHash(table, "#", 0);
    insertHash(table, "R", 1);
    insertHash(table, "H", 2);
    insertHash(table, "K", 3);
    insertHash(table, "D", 4);
    insertHash(table, "E", 5);
    insertHash(table, "S", 6);
    insertHash(table, "T", 7);
    insertHash(table, "N", 8);
    insertHash(table, "Q", 9);
    insertHash(table, "C", 10);
    insertHash(table, "G", 11);
    insertHash(table, "P", 12);
    insertHash(table, "A", 13);
    insertHash(table, "V", 14);
    insertHash(table, "I", 15);
    insertHash(table, "L", 16);
    insertHash(table, "M", 17);
    insertHash(table, "F", 18);
    insertHash(table, "Y", 19);
    insertHash(table, "W", 20);
}


void freeHashTable(HashTable *table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (table->items[i] != NULL) {
            free(table->items[i]->key);
            free(table->items[i]);
        }
    }
}