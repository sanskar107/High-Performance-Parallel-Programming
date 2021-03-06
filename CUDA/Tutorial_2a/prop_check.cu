#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void  printDevProp(cudaDeviceProp  devp)
{

	printf("No. of  multiprocessors: %d\n", devp.multiProcessorCount); //24
	printf("Size of warp %d\n", devp.warpSize ); //32
	printf("Max threads per block %d\n", devp.maxThreadsPerBlock)
	return;
}

int  main()
{
	std::cout<<"Adasd";
	int  devCount = 4;
	cudaGetDeviceCount (&devCount);
	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp  devp;
		cudaGetDeviceProperties (&devp ,i);
		printDevProp(devp);
	}
	return  0;
}