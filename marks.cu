#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/find.h>
#include <thrust/distance.h>

#define NUM_RECORDS 32768
#define THREADS_PER_BLOCK 256
#define NUMBER_LOOPS 1




//Exercise 1) CUDA Parallel reduction
__global__ void maximumMark_CUDA_kernel(float *d_marks, float *d_reduced_marks) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sdata[THREADS_PER_BLOCK];

	//Load a single student mark
	sdata[threadIdx.x] = d_marks[idx];

	//sync threads required to ensure all threads have finished writing
	__syncthreads();


	
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
		//Exercise 1.1) reduce two values per loop and write these back to shared memory
		//ToDo
		
		
		//sync threads required to ensure all threads have finished writing
		__syncthreads();
	}


	//Write the result to shared memory
	if (threadIdx.x == 0){
		d_reduced_marks[blockIdx.x] = sdata[0];
	}
}

void checkCUDAError(const char*);
void readmarks(float *marks);


void calculate_CPU(float*);
void maximumMark_CUDA(float*, float*);
void maximumMark_Thrust(float*);
int sortSplit_Thrust(float*);
int partition_Thrust(float *);



int main(void) {
	//function variables
	float *h_marks;
	float *h_marks_temp;
	cudaEvent_t start, stop;
	float milliseconds = 0;
	int index = 0;

	//create some events for CUDA timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//host allocation
	h_marks = (float*)malloc(sizeof(float)*NUM_RECORDS);
	h_marks_temp = (float*)malloc(sizeof(float)*NUM_RECORDS);

	//read file
	readmarks(h_marks);

	//find highest mark and marks higher than 90% (CPU)
	calculate_CPU(h_marks);

	//Exercise 1) find highest mark with shared memory
	maximumMark_CUDA(h_marks, h_marks_temp);

	//find highest mark (Thrust)
	maximumMark_Thrust(h_marks);

	//find number of marks greater than 90% using a sort and find
	cudaEventRecord(start);
	for (int i = 0; i < NUMBER_LOOPS; i++){
		index = sortSplit_Thrust(h_marks);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("There are %d marks greater than 90%% (using sort+split)\n",index);
	printf("\t took %f ms to calculate\n", milliseconds);

	//find number of marks greater than 90% using partition (prefix sum and re-order)
	cudaEventRecord(start);
	for (int i = 0; i < NUMBER_LOOPS; i++){
		index = partition_Thrust(h_marks);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("There are %d marks greater than 90%% (using partition)\n", index);
	printf("\t took %f ms to calculate\n", milliseconds);
	


	// Cleanup
	free(h_marks);
	free(h_marks_temp);



	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void readmarks(float *marks){
	FILE *f = NULL;


	f = fopen("marks.dat", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find marks.dat file \n");
		exit(1);
	}

	//read student marks
	if (fread(marks, sizeof(float), NUM_RECORDS, f) != NUM_RECORDS){
		fprintf(stderr, "Error: Unexpected end of file!\n");
		exit(1);
	}


	fclose(f);
}


void calculate_CPU(float *h_marks){
	unsigned int i;
	float max_mark;
	int count;

	max_mark = 0;
	count = 0;

	//iterate marks on CPU and record highest mark and student id
	for (i = 0; i < NUM_RECORDS; i++){
		float mark = h_marks[i];
		if (mark > max_mark){
			max_mark = mark;
		}
		if (mark > 90.0f)
			count++;
	}

	//output result
	printf("CPU: Highest mark recorded %f there are %d marks greater 90%%\n", max_mark, count);
}

void maximumMark_CUDA(float *h_marks, float *h_marks_temp){
	unsigned int i;
	float *d_marks;
	float *d_marks_temp;
	float max_mark;

	//device allocation
	cudaMalloc((void**)&d_marks, sizeof(float)*NUM_RECORDS);
	cudaMalloc((void**)&d_marks_temp, sizeof(float)*NUM_RECORDS);
	checkCUDAError("CUDA malloc");

	
	//memory copy marks to device
	cudaMemcpy(d_marks, h_marks, sizeof(float)*NUM_RECORDS, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA: CUDA memcpy");

	//Call the shared memory reduction kernel
	dim3 blocksPerGrid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	maximumMark_CUDA_kernel << <blocksPerGrid, threadsPerBlock >> >(d_marks, d_marks_temp);
	cudaDeviceSynchronize();
	checkCUDAError("CUDA: CUDA kernel");


	//Exercise 1.2) copy result of block level reduction back to host and reduce these values serially
	max_mark = 0;
	//ToDo


	//output result
	printf("CUDA: Highest mark recorded %f\n", max_mark);

	//cleanup
	cudaFree(d_marks);
	cudaFree(d_marks_temp);
	checkCUDAError("CUDA cleanup");
}

struct my_maximum
{
	__host__ __device__ float operator()(const float x, const float y)
	{
		return x < y ? y : x;
	}
};

//Exercise 1.3)
void maximumMark_Thrust(float *h_marks){
	float max_mark;

	//Exercise 1.3.1) create a thrust vector

	//Exercise 1.3.2) reduction using max operator

	printf("Thrust: Highest mark recorded %f\n", max_mark);

}



//Exercise 2.1
int sortSplit_Thrust(float *h_marks){
	thrust::device_vector<float>::iterator iter;

	// Exercise 2.1.1)

	//Exercise 2.1.2) sort the marks

	//Exercise 2.1.3) find if greater than 90%

	//Exercise 2.1.4) find index of first 90% mark
	return 0;

}

//Exercise 2.2)
int partition_Thrust(float * h_marks){
	thrust::device_vector<float>::iterator iter;
		
	//create a thrust vector

	//use a partition to apply a prefix sum and re-order

	//find index of first 90% mark
	return 0;

}


