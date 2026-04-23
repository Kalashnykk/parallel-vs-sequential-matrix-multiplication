// A comparison of parallel and sequential matrix multiplication runtimes using MPI in C 

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "mpi.h"

// Size of the matrix (NxN)
#define N 1000
#define MINRANGE 1
#define MAXRANGE 10 
#define ALGO_COUNT 6
typedef enum {
    ALGO_IJK,
    ALGO_IKJ,
    ALGO_JIK,
    ALGO_JKI,
    ALGO_KIJ,
    ALGO_KJI
} Algorithm;
MPI_Status status;

// Whether to print the matrix when completed
bool printResults = false;
bool testSequential = true;
// run the parallel algorithm?
bool testParallel = true;

// Print matrix function declaration
void printMatrix(float matrix[N][N]);
void multiplyMatrixChunk(int M, float matrix1[M][N], float matrix2[N][N], float productMatrix[M][N], Algorithm algo);
void multiplyMatrix(float matrix1[N][N], float matrix2[N][N], float productMatrix[N][N], Algorithm algo);
void zeroMatrix(float matrix[N][N]);
void printResultsTable(double seqTimes[ALGO_COUNT], double parTimes[ALGO_COUNT], bool testSequential, bool testParallel);

// Define matrices
float matrix1[N][N];
float matrix2[N][N];
float productMatrix[N][N];
float sequentialProductMatrix[N][N];

// Counter variables
int i, j, k;

int main(int argc, char **argv)
{
    int numberOfProcessors;
    int processorRank;
    int numberOfWorkers;

    // Processor sending data
    int sourceProcessor;

    // Processor to receive data
    int destinationProcessor;

    // The number of rows for a worker processor to process
    int rows;

    // The subset of a matrix to be processed by workers
    int matrixSubset;

    // Timer variables
    double runTime;
    clock_t begin, end;
    clock_t beginSeq, endSeq;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Parse command line arguments
    for (int arg = 1; arg < argc; arg++)
    {
        if (strcmp(argv[arg], "--print") == 0)
            printResults = true;
        else if (strcmp(argv[arg], "--no-seq") == 0)
            testSequential = false;
        else if (strcmp(argv[arg], "--no-par") == 0)
            testParallel = false;
    }

    // determine MPI rank/size before validating flags so every process can exit cleanly

    // Determine number of processors available
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

    // Determine rank of calling process
    MPI_Comm_rank(MPI_COMM_WORLD, &processorRank);

    numberOfWorkers = numberOfProcessors - 1;

    if (!testParallel && !testSequential) {
        if (processorRank == 0)
            fprintf(stderr, "Error: cannot disable both parallel and sequential tests\n");
        MPI_Finalize();
        return 1;
    }

    // Initialize the product matrix to zero
    zeroMatrix(productMatrix);
    if (testSequential)
        zeroMatrix(sequentialProductMatrix);

    /* ---------- Manager Processor Code ---------- */

    if (processorRank == 0)
    {
        // Populate the matrices with values (needed for either test)
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                matrix1[i][j] = (float)((rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE);
                matrix2[i][j] = (float)((rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE);
            }
        }

        double parTimes[ALGO_COUNT] = {0};
        if (testParallel) {
            printf("\nMultiplication of %dx%d float matrices using %d processor(s).\n", N, N, numberOfProcessors);
            rows = N / numberOfWorkers;

            for (int algo = 0; algo < ALGO_COUNT; algo++) {
                matrixSubset = 0;
                zeroMatrix(productMatrix);
                begin = clock();

                for (destinationProcessor = 1; destinationProcessor <= numberOfWorkers; destinationProcessor++)
                {
                    MPI_Send(&matrixSubset, 1, MPI_INT, destinationProcessor, 1, MPI_COMM_WORLD);
                    MPI_Send(&rows, 1, MPI_INT, destinationProcessor, 1, MPI_COMM_WORLD);
                    MPI_Send(&algo, 1, MPI_INT, destinationProcessor, 1, MPI_COMM_WORLD);
                    MPI_Send(&matrix1[matrixSubset][0], rows * N, MPI_FLOAT, destinationProcessor, 1, MPI_COMM_WORLD);
                    MPI_Send(&matrix2, N * N, MPI_FLOAT, destinationProcessor, 1, MPI_COMM_WORLD);
                    matrixSubset += rows;
                }

                for (i = 1; i <= numberOfWorkers; i++)
                {
                    sourceProcessor = i;
                    MPI_Recv(&matrixSubset, 1, MPI_INT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
                    MPI_Recv(&rows, 1, MPI_INT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
                    MPI_Recv(&productMatrix[matrixSubset][0], rows * N, MPI_FLOAT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
                }

                end = clock();
                parTimes[algo] = (double)(end - begin) / CLOCKS_PER_SEC;
            }
        }
        
        // Sequential matrix multiplication for comparison
        double seqTimes[ALGO_COUNT] = {0};
        if (testSequential == true)
        {
            for (int algo = 0; algo < ALGO_COUNT; algo++) {
                zeroMatrix(sequentialProductMatrix);  // Reset for each run
                clock_t beginSeq = clock();
                multiplyMatrix(matrix1, matrix2, sequentialProductMatrix, (Algorithm)algo);
                clock_t endSeq = clock();
                seqTimes[algo] = (double)(endSeq - beginSeq) / CLOCKS_PER_SEC;
            }
        }
        
        // Optionally print matrix results
        if (printResults == true)
        {
            printf("Matrix 1:\n");
            printMatrix(matrix1);
            printf("Matrix 2:\n");
            printMatrix(matrix2);
            printf("Product Matrix:\n");
            if (testParallel)
                printMatrix(productMatrix);
            else if (testSequential)
                printMatrix(sequentialProductMatrix);
        }
        
        // Print the results table
        printResultsTable(seqTimes, parTimes, testSequential, testParallel);
    }

    /* ---------- Worker Processor Code ---------- */

    if (processorRank > 0)
    {
        if (!testParallel) {
            MPI_Finalize();
            return 0;
        }

        for (int algo = 0; algo < ALGO_COUNT; algo++) {
            int algoId;
            sourceProcessor = 0;
            MPI_Recv(&matrixSubset, 1, MPI_INT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&algoId, 1, MPI_INT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix1, rows * N, MPI_FLOAT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix2, N * N, MPI_FLOAT, sourceProcessor, 1, MPI_COMM_WORLD, &status);

            /* Perform matrix multiplication */
            zeroMatrix(productMatrix);
            multiplyMatrixChunk(rows, matrix1, matrix2, productMatrix, (Algorithm)algoId);

            MPI_Send(&matrixSubset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&productMatrix, rows * N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}

/**
 * @brief Prints the contents of an NxN matrix
 * 
 * @param matrix An NxN matrix of floats
 */
void printMatrix(float matrix[N][N])
{
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Multiplies a chunk of the first matrix with the second matrix and stores the result in the product matrix
 * 
 * @param M Number of rows in the chunk
 * @param matrix1 First matrix chunk
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 * @param algo Algorithm to use for multiplication
 */
void multiplyMatrixChunk(int M, float matrix1[M][N], float matrix2[N][N], float productMatrix[M][N], Algorithm algo) 
{
    switch (algo) {
        case ALGO_IJK:
            for (i = 0; i < M; i++)
                for (j = 0; j < N; j++)
                    for (k = 0; k < N; k++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
        case ALGO_IKJ:
            for (i = 0; i < M; i++)
                for (k = 0; k < N; k++)
                    for (j = 0; j < N; j++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
        case ALGO_JIK:
            for (j = 0; j < N; j++)
                for (i = 0; i < M; i++)
                    for (k = 0; k < N; k++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
        case ALGO_JKI:
            for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                    for (i = 0; i < M; i++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
        case ALGO_KIJ:
            for (k = 0; k < N; k++)
                for (i = 0; i < M; i++)
                    for (j = 0; j < N; j++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
        case ALGO_KJI:
            for (k = 0; k < N; k++)
                for (j = 0; j < N; j++)
                    for (i = 0; i < M; i++)
                        productMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            break;
    }
}

/**
 * @brief Multiplies two NxN matrices and stores the result in the product matrix
 * 
 * @param matrix1 First matrix
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 * @param algo Algorithm to use for multiplication
 */
void multiplyMatrix(float matrix1[N][N], float matrix2[N][N], float productMatrix[N][N], Algorithm algo) 
{
    multiplyMatrixChunk(N, matrix1, matrix2, productMatrix, algo);
}

/**
 * @brief Sets all values in an NxN matrix to zero
 * 
 * @param matrix An NxN matrix of floats
 */
void zeroMatrix(float matrix[N][N]) 
{
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0f;
        }
    }
}

/**
 * @brief Prints the execution times in a table format
 * 
 * @param seqTimes Array of sequential times for each algorithm
 * @param parTimes Parallel times array (if tested)
 * @param testSequential Whether sequential was tested
 * @param testParallel Whether parallel was tested
 */
void printResultsTable(double seqTimes[ALGO_COUNT], double parTimes[ALGO_COUNT], bool testSequential, bool testParallel)
{
    const char* algoNames[ALGO_COUNT] = {"IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"};
    
    printf("\nExecution Times (seconds):\n");
    printf("---------------------------------\n");
    printf("Algorithm | Sequential | Parallel\n");
    printf("----------|------------|---------\n");
    
    for (int a = 0; a < ALGO_COUNT; a++) {
        printf("   %-6s | ", algoNames[a]);
        if (testSequential) {
            printf("%10.6f | ", seqTimes[a]);
        } else {
            printf("     N/A    | ");
        }
        if (testParallel) {
            printf("%7.6f\n", parTimes[a]);
        } else {
            printf("  N/A\n");
        }
    }

    printf("---------------------------------\n");
}
