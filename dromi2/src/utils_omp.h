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
void norm2(const double* array, int length, double* val);
void printArray(const double* array, int N);
void cosine_sim(const double* arrayA, const double* arrayB, int length, double* similarity);
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