// Test program to compare three matrix multiplication functions with different loop orders

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

// Size of the matrix (NxN)
#define N 1500
#define MINRANGE 1
#define MAXRANGE 10

// Whether to print the matrices
bool printResults = false;

// Define matrices
int matrix1[N][N];
int matrix2[N][N];
int productMatrixKIJ[N][N];
int productMatrixIJK[N][N];
int productMatrixIKJ[N][N];

// Counter variables
int i, j, k;

// Function declarations
void printMatrix(int matrix[N][N]);
void multiplyMatrixChunkKIJ(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]);
void multiplyMatrixChunkIJK(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]);
void multiplyMatrixChunkIKJ(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]);
void zeroMatrix(int matrix[N][N]);

int main(int argc, char **argv)
{
    // Timer variables
    double runTime;
    clock_t begin, end;
    
    // Initialize matrices
    zeroMatrix(productMatrixKIJ);
    zeroMatrix(productMatrixIJK);
    zeroMatrix(productMatrixIKJ);

    printf("\nTesting three matrix multiplication variants with %dx%d matrices.\n\n", N, N);

    // Populate the matrices with values
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            matrix1[i][j] = (rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE;
            matrix2[i][j] = (rand() % (MAXRANGE - MINRANGE + 1)) + MINRANGE;
        }
    }

    // Print input matrices
    if (printResults == true)
    {
        printf("Matrix 1:\n");
        printMatrix(matrix1);
        printf("Matrix 2:\n");
        printMatrix(matrix2);
    }

    // Test KIJ loop order
    printf("Testing multiplyMatrixChunkKIJ (KIJ loop order)...\n");
    begin = clock();
    multiplyMatrixChunkKIJ(N, matrix1, matrix2, productMatrixKIJ);
    end = clock();
    runTime = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", runTime);
    if (printResults == true)
    {
        printf("Product Matrix (KIJ):\n");
        printMatrix(productMatrixKIJ);
    }

    // Test IJK loop order
    printf("\nTesting multiplyMatrixChunkIJK (IJK loop order)...\n");
    begin = clock();
    multiplyMatrixChunkIJK(N, matrix1, matrix2, productMatrixIJK);
    end = clock();
    runTime = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", runTime);
    if (printResults == true)
    {
        printf("Product Matrix (IJK):\n");
        printMatrix(productMatrixIJK);
    }

    // Test IKJ loop order
    printf("\nTesting multiplyMatrixChunkIKJ (IKJ loop order)...\n");
    begin = clock();
    multiplyMatrixChunkIKJ(N, matrix1, matrix2, productMatrixIKJ);
    end = clock();
    runTime = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", runTime);
    if (printResults == true)
    {
        printf("Product Matrix (IKJ):\n");
        printMatrix(productMatrixIKJ);
    }

    // Verify all results are identical
    printf("\nVerifying correctness...\n");
    bool identical = true;
    for (i = 0; i < N && identical; i++)
    {
        for (j = 0; j < N && identical; j++)
        {
            if (productMatrixKIJ[i][j] != productMatrixIJK[i][j] || productMatrixKIJ[i][j] != productMatrixIKJ[i][j])
            {
                identical = false;
            }
        }
    }
    
    if (identical)
    {
        printf("All matrices are identical - CORRECT!\n\n");
    }
    else
    {
        printf("Matrices differ - ERROR!\n\n");
    }

    return 0;
}

/**
 * @brief Prints the contents of an NxN matrix
 * 
 * @param matrix An NxN matrix of integers
 */
void printMatrix(int matrix[N][N])
{
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Multiplies a chunk of the first matrix with the second matrix (KIJ loop order)
 * 
 * @param M Number of rows in the chunk
 * @param matrix1 First matrix chunk
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 */
void multiplyMatrixChunkKIJ(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]) 
{
    for (k = 0; k < N; k++)
    {
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < N; j++)
            {
                productMatrix[i][k] = productMatrix[i][k] + matrix1[i][j] * matrix2[j][k];
            }
        }
    }
}

/**
 * @brief Multiplies a chunk of the first matrix with the second matrix (IJK loop order)
 * 
 * @param M Number of rows in the chunk
 * @param matrix1 First matrix chunk
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 */
void multiplyMatrixChunkIJK(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]) 
{
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                productMatrix[i][j] = productMatrix[i][j] + matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

/**
 * @brief Multiplies a chunk of the first matrix with the second matrix (IKJ loop order)
 * 
 * @param M Number of rows in the chunk
 * @param matrix1 First matrix chunk
 * @param matrix2 Second matrix
 * @param productMatrix Product matrix
 */
void multiplyMatrixChunkIKJ(int M, int matrix1[M][N], int matrix2[N][N], int productMatrix[M][N]) 
{
    for (i = 0; i < M; i++)
    {
        for (k = 0; k < N; k++)
        {
            for (j = 0; j < N; j++)
            {
                productMatrix[i][j] = productMatrix[i][j] + matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

/**
 * @brief Sets all values in an NxN matrix to zero
 * 
 * @param matrix An NxN matrix of integers
 */
void zeroMatrix(int matrix[N][N]) 
{
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
        }
    }
}
