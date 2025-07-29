// dqn_openmp.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define BATCH 256
#define INPUT_SIZE 128
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 4

// ReLU function
inline float relu(float x) {
    return x > 0 ? x : 0;
}

void forward_pass(const std::vector<float>& input,
                  const std::vector<float>& W1,
                  const std::vector<float>& W2,
                  std::vector<float>& output) {

    std::vector<float> hidden(BATCH * HIDDEN_SIZE);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < BATCH; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            float sum = 0;
            for (int k = 0; k < INPUT_SIZE; ++k)
                sum += input[i * INPUT_SIZE + k] * W1[k * HIDDEN_SIZE + j];
            hidden[i * HIDDEN_SIZE + j] = relu(sum);
        }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < BATCH; ++i)
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float sum = 0;
            for (int k = 0; k < HIDDEN_SIZE; ++k)
                sum += hidden[i * HIDDEN_SIZE + k] * W2[k * OUTPUT_SIZE + j];
            output[i * OUTPUT_SIZE + j] = relu(sum);
        }
}

int main() {
    std::vector<float> input(BATCH * INPUT_SIZE, 1.0f);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE, 0.01f);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE, 0.01f);
    std::vector<float> output(BATCH * OUTPUT_SIZE, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    forward_pass(input, W1, W2, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "OpenMP forward pass time: " << duration.count() << " seconds\n";

    return 0;
}
