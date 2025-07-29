#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define SIZE 1024

void matrixMultiplyOpenMP(const std::vector<std::vector<float>> &A,
                          const std::vector<std::vector<float>> &B,
                          std::vector<std::vector<float>> &C) {
    #pragma omp parallel
    {
        // Print from a single thread to avoid clutter
        #pragma omp single
        std::cout << "Using " << omp_get_num_threads() << " threads\n";
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

int main() {
    std::vector<std::vector<float>> A(SIZE, std::vector<float>(SIZE, 1.0));
    std::vector<std::vector<float>> B(SIZE, std::vector<float>(SIZE, 2.0));
    std::vector<std::vector<float>> C(SIZE, std::vector<float>(SIZE, 0.0));

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyOpenMP(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "OpenMP Time: " << duration.count() << " seconds\n";

    return 0;
}
