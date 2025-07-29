# OpenMP-CUDA
Benchmark the performance of CPU-based parallelism against GPU computing for simple ML related tasks

## Overview

This project provides implementations of matrix multiplication and a simple Deep Q-Network (DQN) forward pass using OpenMP and CUDA.

## Requirements

- g++ with OpenMP support
- NVIDIA CUDA Toolkit
- Python 3.x with matplotlib

## Setup, compilation and usage

Compile OpenMP version:

`g++ -fopenmp -O3 Folder/example.cpp -o Folder/example`

Compile CUDA version:

`nvcc -O3 Folder/example.cu -o Folder/example`

Run benchmark script:

`python3 benchmark_example.py`
