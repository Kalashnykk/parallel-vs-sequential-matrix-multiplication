#include "mpi.h"
#include <stdio.h>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int numberOfProcessors, processorRank;

    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &processorRank);

    printf("Hello from the processor (rank %d out of %d processors)\n", processorRank, numberOfProcessors);

    MPI_Finalize();
}