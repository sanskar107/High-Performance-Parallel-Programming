#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swap(int*, int);
__global__ void reflect(int *M, int mat_size);

