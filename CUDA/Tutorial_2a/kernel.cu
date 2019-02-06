
__global__ void swap(int *M, int mat_size)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int N = mat_size;

	if((i < N) && (j < N) && (j%2 == 0) && (j != N - 1))
	{
		int tmp = M[i * mat_size + j];
		M[i * mat_size + j] = M[i*mat_size + j + 1];
		M[i * mat_size + j + 1] = tmp;
	}
	__syncthreads();
}

