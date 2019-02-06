__global__ void process_kernel1(const float *input1,const  float *input2, float *output, int datasize)
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;

	if(i < datasize)
	{
		output[i] = sin(input1[i]) + cos(input2[i]);
	}
}
	

__global__ void process_kernel2(const float *input, float *output, int datasize)
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;
	if(i < datasize)
	{
		output[i] = log(input[i]);
	}
}

__global__ void process_kernel3(const float *input, float *output, int datasize)
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;
	if(i < datasize)
	{
		output[i] = sqrt(input[i]);
	}
}
