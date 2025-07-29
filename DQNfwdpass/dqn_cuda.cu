// dqn_cuda.cu
#include <iostream>
#include <cuda_runtime.h>

#define BATCH 256
#define INPUT_SIZE 128
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 4

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void forward_hidden(const float* input, const float* W1, float* hidden) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    float sum = 0;
    for (int k = 0; k < INPUT_SIZE; ++k)
        sum += input[i * INPUT_SIZE + k] * W1[k * HIDDEN_SIZE + j];
    hidden[i * HIDDEN_SIZE + j] = relu(sum);
}

__global__ void forward_output(const float* hidden, const float* W2, float* output) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    float sum = 0;
    for (int k = 0; k < HIDDEN_SIZE; ++k)
        sum += hidden[i * HIDDEN_SIZE + k] * W2[k * OUTPUT_SIZE + j];
    output[i * OUTPUT_SIZE + j] = relu(sum);
}

int main() {
    size_t in_size = BATCH * INPUT_SIZE * sizeof(float);
    size_t h1_size = INPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t h2_size = HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t hid_size = BATCH * HIDDEN_SIZE * sizeof(float);
    size_t out_size = BATCH * OUTPUT_SIZE * sizeof(float);

    float *h_input = new float[BATCH * INPUT_SIZE];
    float *h_W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *h_W2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float *h_output = new float[BATCH * OUTPUT_SIZE];

    std::fill(h_input, h_input + BATCH * INPUT_SIZE, 1.0f);
    std::fill(h_W1, h_W1 + INPUT_SIZE * HIDDEN_SIZE, 0.01f);
    std::fill(h_W2, h_W2 + HIDDEN_SIZE * OUTPUT_SIZE, 0.01f);

    float *d_input, *d_W1, *d_W2, *d_hidden, *d_output;
    cudaMalloc(&d_input, in_size);
    cudaMalloc(&d_W1, h1_size);
    cudaMalloc(&d_W2, h2_size);
    cudaMalloc(&d_hidden, hid_size);
    cudaMalloc(&d_output, out_size);

    cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, h1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, h2_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    forward_hidden<<<BATCH, HIDDEN_SIZE>>>(d_input, d_W1, d_hidden);
    forward_output<<<BATCH, OUTPUT_SIZE>>>(d_hidden, d_W2, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA forward pass time: " << milliseconds / 1000.0 << " seconds\n";

    cudaFree(d_input); cudaFree(d_W1); cudaFree(d_W2);
    cudaFree(d_hidden); cudaFree(d_output);
    delete[] h_input; delete[] h_W1; delete[] h_W2; delete[] h_output;

    return 0;
}
