#ifndef UTILS_H
#define UTILS_H

// Define the structs


// Define the structure for each item in the hash table (key-value pair)
typedef struct {
    char *key;
    int value;
} HashItem;

// Define the structure for the hash table
typedef struct {
    HashItem *items[64];  // Array of pointers to HashItems
} HashTable;

// Functions
void norm2(const float* array, int length, float* val);
void printArray(const float* array, int N);
// void cosine_sim(const float* arrayA, const float* arrayB, int length, float* similarity);
float cosine_sim(float* arrayA, float* arrayB, int length);
void cosine_sim_2d(
    float** matrixA,
    float** matrixB,
    float** result_matrix,
    int num_arrays,
    int length_arrays
    );
void cosine_sim_3d(
    float*** tensorA,
    float*** tensorB,
    float** result_matrix,
    int num_arrays,
    int length_arrays,
    int length_features
    );
void cosine_sim_3d_masked(
    float*** tensorA,
    float*** tensorB,
    float** result_matrix,
    int num_arrays,
    int length_arrays,
    int length_features
    );
void load_tensor_binary(
    const char *filename,
    float ****tensor,
    int *x_dim,
    int *y_dim,
    int *z_dim
    );
void readFileAndStoreStrings(const char *filename, char ***strings, int *numStrings, int maxLength);
void padStringsToMaxLength(char **strings, int numStrings, int maxLength);
void freeStrings(char **strings, int numStrings);
unsigned int hash(const char *key);
void initHashTable(HashTable *table);
void insertHash(HashTable *table, const char *key, int value);
int getHash(HashTable *table, const char *key);
void hashAminoAcids(HashTable *table);
void freeHashTable(HashTable *table);
#endif // UTILS_H