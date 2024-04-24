#include<iostream>
#include<cuda_runtime.h>
using namespace std;

// defining the CUDA kernel function which will be executed on the GPU
__global__ void matmul(int* A, int* B, int* C, int N)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if(Row < N && Col < N) 
    {
        int Pvalue = 0;
        for(int k=0; k<N; k++)
        {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

int main() {
    int N = 512; // size of the matrix
    int size = N * N * sizeof(int);
    int *A;
    int *B;
    int *C;

    // Allocate memory on the host
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize matrices A and B
    for(int i=0; i<N; i++) 
    {
        for(int j=0; j<N; j++)
        {
            A[i*N+j] = i*N+j;
            B[i*N+j] = j*N+i;
        }
    }

    // Allocate memory on the device
    int *dev_A;
    int *dev_B;
    int *dev_C;
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Copy data from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launching the kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Ensure that the kernel has completed execution before copying the result
    cudaDeviceSynchronize();

    // Copy data from device to the host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print the results
    for(int i=0; i<10; i++)
    {
        for(int j=0; j<10; j++) 
        {
            cout<<C[i*N+j]<<" ";
        }
        cout<<endl;
    }

    // Free memory from the device and the host
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

// Compiling & Running code
// nvcc -o program_name program-name.cu
// ./program_name