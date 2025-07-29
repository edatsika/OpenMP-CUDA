#include <iostream>
#include <cuda_runtime.h>

#define SIZE 1024

__global__ void matrixMultiplyCUDA(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < SIZE && col < SIZE) {
        float sum = 0.0f;
        for (int k = 0; k < SIZE; ++k)
            sum += A[row * SIZE + k] * B[k * SIZE + col];
        C[row * SIZE + col] = sum;
    }
}

int main() {
    int N = SIZE * SIZE;
    size_t bytes = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(SIZE / threads.x, SIZE / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyCUDA<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA Time: " << milliseconds / 1000.0 << " seconds\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}