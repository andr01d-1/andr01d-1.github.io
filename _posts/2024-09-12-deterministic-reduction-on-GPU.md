---
layout: post
title: "Deterministic Reduction and CGA on GPU"
categories: low-level-GPU
---

The two topics are not necessarily related. 


CGA (Cooperative Grid Array) is

> A new cooperative grouping introduced in the Hopper Architecture (SM90)
> [Clusters enable multiple thread blocks running concurrently across multiple SMs to synchronize and collaboratively fetch and exchange data](https://stackoverflow.com/questions/78510678/whats-cga-in-cuda-programming-model)

CoffeeBeforeArch has a great [tutorial](https://www.youtube.com/watch?v=k7K-h7P1Bdk) on this [code sample](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/03_sum_reduction/cooperative_groups/sumReduction.cu)

However, using `atomicAdd` on float-point numbers will introduce nondeterministic results.

Let's design a deterministic reduction algorithm using CGA for float point numbers

```cpp

// cga_deterministic_reduction.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <algorithm>

namespace cg = cooperative_groups;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float blockReduceSum(cg::thread_block_tile<32>& tile, float val) {
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

__global__ void firstPassReduction(float* input, float* output, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (tid < N) {
        sum = input[tid];
    }
    
    // Warp-level reduction
    sum = blockReduceSum(tile, sum);
    
    // Block-level reduction using shared memory
    __shared__ float sdata[32];
    if (tile.thread_rank() == 0) {
        sdata[threadIdx.x / 32] = sum;
    }
    block.sync();
    
    if (threadIdx.x < 32) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : 0.0f;
        sum = blockReduceSum(tile, sum);
    }

    // Store block results
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

__global__ void secondPassReduction(float* input, float* output, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += input[i];
    }
    
    sum = blockReduceSum(tile, sum);
    
    if (threadIdx.x == 0) {
        output[0] = sum;
    }
}

int main() {
    const int N = 1 << 13;  // 8192 elements
    float *h_input, *h_output, *d_input, *d_output;

    // Allocate and initialize host memory
    h_input = new float[N];
    h_output = new float[1];
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Initialize all elements to 1
    }
    *h_output = 0.0f;

    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_output, N * sizeof(float))); // Allocate worst-case size

    // Copy input data to device
    cudaCheckError(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Launching first pass kernel with %d blocks of %d threads each\n", numBlocks, blockSize);

    firstPassReduction<<<numBlocks, blockSize>>>(d_input, d_output, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Debug: Print partial sums
    float* h_partial_sums = new float[numBlocks];
    cudaMemcpy(h_partial_sums, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float cpu_sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        cpu_sum += h_partial_sums[i];
        if (i < 10) printf("Partial sum %d: %f\n", i, h_partial_sums[i]);
    }
    printf("Sum of partial sums on CPU: %f\n", cpu_sum);
    printf("First partial sum: %f\n", h_partial_sums[0]);
    printf("Last partial sum: %f\n", h_partial_sums[numBlocks-1]);
    delete[] h_partial_sums;

    // Launch second pass kernel
    printf("Launching second pass kernel with 1 block of %d threads\n", blockSize);

    secondPassReduction<<<1, blockSize>>>(d_output, d_output, numBlocks);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum from GPU: %f\n", *h_output);
    printf("Expected sum: %f\n", (float)N);

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

```


```bash
nvcc --arch=sm_90 cga_deterministic_reduction.cu -o deterministic_reduction
```

**What is `thread_rank()`**

- Returns the rank (index) of the calling thread within its group

- The rank is an integer value between 0 and size() - 1, where size() is the total number of threads in the group

- It is available as a method on various group types, including:
  - thread_group
  - thread_block
  - coalesced_group
  - multi_grid_group

- The rank provides a unique identifier for each thread within its group, which is useful for 
  - Distributing work among threads
  - Determining which thread should perform certain operations
  - Indexing into shared resources
- In cooperative groups, `thread_rank()` is more flexible than traditional CUDA thread indexing (like threadIdx.x) because it works across different types of thread groups,
  not just within a single block
- The ranking is based on the group's organization, so for coalesced groups, the rank might not correspond directly to the thread's position in the original block or grid

```cpp
shfl_down
```

`__shfl_down` is particularly useful for optimizing algorithms that require communication between nearby threads, such as parallel reductions or scan operations, by leveraging the [warp's SIMD execution model](https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf)