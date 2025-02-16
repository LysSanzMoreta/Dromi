#ifndef UTILS_H
#define UTILS_H

// Define the structs
#define TABLE_SIZE 64

// Define the structure for each item in the hash table (key-value pair)
typedef struct HashItem{
    char *key;
    float *value;
    int length;
} HashItem;

// Define the structure for the hash table
typedef struct HashTable{
    HashItem *items[TABLE_SIZE];  // Array of pointers to HashItems
} HashTable;

// Functions
void norm2(const float* array, int length, float* val);
void printArray(const float* array, int N);
void printCharArray(const char* array, int N);
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
void save_matrix_binary(
    const char *filename,
    float ***matrix,
    int x_dim,
    int y_dim
);
void readFileAndStoreStrings(const char *filename, char ***strings, int *numStrings, int maxLength);
void padStringsToMaxLength(char **strings, int numStrings, int maxLength);
void freeStrings(char **strings, int numStrings);
unsigned int hash(const char *key);
void initHashTable(HashTable *table);
// void printHashItem(HashItem *item);
// void insertHash(HashTable *table, const char *key, int value);
void insertHash(HashTable *table, const char *key, float* value, int length);
float* getHash(HashTable *table, const char *key);
void hashAminoAcids(HashTable *table, const int seed, const int length);
void freeHashTable(HashTable *table);
void getZeros(int length, float* array);
void zeroOutTensor(float*** tensor, int x, int y, int z);
#endif // UTILS_H