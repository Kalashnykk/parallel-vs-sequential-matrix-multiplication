// A comparison of parallel and sequential matrix multiplication runtimes using MPI in C 

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "mpi.h"

// Size of the matrix (NxN)
#define N 2000
#define MINRANGE 1
#define MAXRANGE 10 

MPI_Status status;

// Whether to print the matrix when completed
bool printResults = false;
bool testSequential = true;

// Print matrix function declaration
void printMatrix(float matrix[N][N]);
void multiplyMatrixChunk(int M, float matrix1[M][N], float matrix2[N][N], float productMatrix[M][N]);
void muptiplyMatrix(float matrix1[N][N], float matrix2[N][N], float productMatrix[N][N]);
void zeroMatrix(float matrix[N][N]);

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
    }

    // Determine number of processors available
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

    // Determine rank of calling process
    MPI_Comm_rank(MPI_COMM_WORLD, &processorRank);

    numberOfWorkers = numberOfProcessors - 1;

    // Initialize the product matrix to zero
    zeroMatrix(productMatrix);
    testSequential ? (zeroMatrix(sequentialProductMatrix)) : (0);

    /* ---------- Manager Processor Code ---------- */

    if (processorRank == 0)
    {
        // Initialize a timer
        begin = clock();

        printf("\nMultiplication of %dx%d float matrices using %d processor(s) has been started.\n\n", N, N, numberOfProcessors);

        // Populate the matrices with values
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                matrix1[i][j] = (float)((rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE);
                matrix2[i][j] = (float)((rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE);
            }
        }

        /* Send the matrix to the worker processes */
        rows = N / numberOfWorkers;
        matrixSubset = 0;

        // Iterate through all of the workers and assign work
        for (destinationProcessor = 1; destinationProcessor <= numberOfWorkers; destinationProcessor++)
        {
            // Determine the subset of the matrix to send to the destination processor
            MPI_Send(&matrixSubset, 1, MPI_INT, destinationProcessor, 1, MPI_COMM_WORLD);

            // Send the number of rows to process to the destination worker processor
            MPI_Send(&rows, 1, MPI_INT, destinationProcessor, 1, MPI_COMM_WORLD);

            // Send rows from matrix 1 to destination worker processor
            MPI_Send(&matrix1[matrixSubset][0], rows * N, MPI_FLOAT, destinationProcessor, 1, MPI_COMM_WORLD);

            // Send entire matrix 2 to destination worker processor
            MPI_Send(&matrix2, N * N, MPI_FLOAT, destinationProcessor, 1, MPI_COMM_WORLD);

            // Determine the next chunk of data to send to the next processor
            matrixSubset = matrixSubset + rows;
        }

        // Retrieve results from all workers processors
        for (i = 1; i <= numberOfWorkers; i++)
        {
            sourceProcessor = i;
            MPI_Recv(&matrixSubset, 1, MPI_INT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&productMatrix[matrixSubset][0], rows * N, MPI_FLOAT, sourceProcessor, 2, MPI_COMM_WORLD, &status);
        }

        // Stop the timer
        end = clock();
        
        // Sequential matrix multiplication for comparison
        if (testSequential == true)
        {
            // Restart the timer for the sequential matrix multiplication
            beginSeq = clock();

            // Perform sequential matrix multiplication
            muptiplyMatrix(matrix1, matrix2, sequentialProductMatrix);
            
            // Stop the timer
            endSeq = clock();
        }
        
        // Optionally print matrix results
        if (printResults == true)
        {
            printf("Matrix 1:\n");
            printMatrix(matrix1);
            printf("Matrix 2:\n");
            printMatrix(matrix2);
            printf("Product Matrix:\n");
            printMatrix(productMatrix);            
        }
        printf("Multiplication of %dx%d float matrices using %d processor(s) has been completed.\n\n", N, N, numberOfProcessors);

        // Determine and print runtimes for parallel and sequential matrix multiplication
        if (testSequential == true) 
        {
            printf("Runtimes for parallel and sequential matrix multiplication:\n");

            runTime = (double)(endSeq - beginSeq) / CLOCKS_PER_SEC;
            printf("Sequential (s): %f\n", runTime);
        }
        
        runTime = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("Parallel (s):   %f\n\n", runTime);
    }

    /* ---------- Worker Processor Code ---------- */

    if (processorRank > 0)
    {
        sourceProcessor = 0;
        MPI_Recv(&matrixSubset, 1, MPI_INT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix1, rows * N, MPI_FLOAT, sourceProcessor, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix2, N * N, MPI_FLOAT, sourceProcessor, 1, MPI_COMM_WORLD, &status);

        /* Perform matrix multiplication */

        multiplyMatrixChunk(rows, matrix1, matrix2, productMatrix);

        MPI_Send(&matrixSubset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&productMatrix, rows * N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
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
 */
void multiplyMatrixChunk(int M, float matrix1[M][N], float matrix2[N][N], float productMatrix[M][N]) 
{
    for (i = 0; i < M; i++)
    {
        for (k = 0; k < N; k++)
        {
            for (j = 0; j < N; j++)
            {
                productMatrix[i][k] = productMatrix[i][k] + matrix1[i][j] * matrix2[j][k];
            }
        }
    }
}

/**
 * @brief Multiplies two NxN matrices and stores the result in the product matrix
 * 
 * @param matrix1 First matrix
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 */
void muptiplyMatrix(float matrix1[N][N], float matrix2[N][N], float productMatrix[N][N]) 
{
    multiplyMatrixChunk(N, matrix1, matrix2, productMatrix);
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
