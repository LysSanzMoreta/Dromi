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

void printCharArray(const char* array, int N) {
    for (int i = 0; i < N; ++i) {
        printf("%c ", array[i]);
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
    int x_dim,
    int y_dim
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
    size_t matrix_size = x_dim * y_dim ;
    fwrite(matrix, sizeof(float), matrix_size, file);

    fclose(file);
}


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


// void printHashItem(HashItem *item) {
//     if (item) {
//         printf("Key: %s\n", item->key);
//         printf("Value (Random Floats): ");
//         for (int i = 0; i < item->length; ++i) {
//             printf("%f ", item->value[i]);
//         }
//         printf("\n");
//     }
// }


// Function to insert a new item into the hash table
void insertHash(HashTable *table, const char *key, float* value, int length) {
    unsigned int index = hash(key);  // Calculate index using hash function

    // Create a new HashItem and populate it with key and random float array
    HashItem *newItem = (HashItem *)malloc(sizeof(HashItem));
    newItem->key = strdup(key);  // Copy the key string

    // Allocate memory for the float array and copy the values
    newItem->value = (float *)malloc(sizeof(float) * length);
    for (int i = 0; i < length; ++i) {
        newItem->value[i] = value[i];  // Copy each float from the provided array
    }

    newItem->length = length;  // Store the length of the array

    // Insert the new item into the hash table at the computed index
    table->items[index] = newItem;
}

// void insertHash(HashTable *table, const char *key, int value) {
//     unsigned int index = hash(key);
//     HashItem *newItem = (HashItem *)malloc(sizeof(HashItem));
//     newItem->key = strdup(key);  // Copy the key string
//     newItem->value = value;
//     table->items[index] = newItem;
// }


float* getHash(HashTable *table, const char *key) {
    unsigned int index = hash(key);
    float* not_found = NULL;
    if (table->items[index] != NULL && strcmp(table->items[index]->key, key) == 0) {
        // printf("hash val: %\n", table->items[index]->value);
        return table->items[index]->value;
    } else {
        return not_found;  // Key not found, return -1
    }
}

void generateRandomFloats(int seed, int length, float* array) {
    // Seed the random number generator with the provided seed
    srand(seed);

    // Generate random floats between 0 and 1
    #pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        array[i] = (float)rand() / RAND_MAX;  // Scale rand() to the range [0, 1]
    }
}


void getZeros(int length, float* array) {
    #pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        array[i] = 0.0;
    }
}


void zeroOutTensor(float*** tensor, int x, int y, int z){
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                tensor[i][j][k] = 0.0;
            }

        }
    }
}


void hashAminoAcids(HashTable *table, const int seed, const int length){
    // Insert the amino acid mappings into the hash table
    float* randomArray = (float*)malloc(length * sizeof(float));

    // Insert 0's for #
    getZeros(length, randomArray);
    // generateRandomFloats(seed, length, randomArray);
    insertHash(table, "#", randomArray, length);

    const char* keys[20] = {
        "R",
        "H",
        "K",
        "D",
        "E",
        "S",
        "T",
        "N",
        "Q",
        "C",
        "G",
        "P",
        "A",
        "V",
        "I",
        "L",
        "M",
        "F",
        "Y",
        "W"
    };
    for (int i = 0; i < 20; ++i) {
        generateRandomFloats(seed+i, length, randomArray);
        insertHash(table, keys[i], randomArray, length);
    }
}


void freeHashTable(HashTable *table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (table->items[i] != NULL) {
            free(table->items[i]->key);
            free(table->items[i]->value);
            free(table->items[i]);
        }
    }
}