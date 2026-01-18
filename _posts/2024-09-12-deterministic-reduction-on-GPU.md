---
layout: post
title: "Deterministic Reduction and CGA on GPU"
categories: low-level-GPU
---


<!-- This is one series in CUDA basics

sum reduction is 


VLSI adder/sum tree 

shuffle command 

and 


## Parallel Algorithm


Scan is a  -->

## Why would we need [Cooperative Groups (CG)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

- Finer-Grained Synchronization: Move beyond the *block-level __syncthreads()* to synchronize smaller groups (like *warps*) or partitions within a block,
  Essential for algorithms requiring tighter coordination


CGA (Cooperative Grid Array) is

> A new cooperative grouping introduced in the Hopper Architecture (SM90)
> [Clusters enable multiple thread blocks running concurrently across multiple SMs to synchronize and collaboratively fetch and exchange data](https://stackoverflow.com/questions/78510678/whats-cga-in-cuda-programming-model)

CoffeeBeforeArch has a great [tutorial](https://www.youtube.com/watch?v=k7K-h7P1Bdk) on this [code sample](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/03_sum_reduction/cooperative_groups/sumReduction.cu)

However, using `atomicAdd` on float-point numbers will introduce non-deterministic results.

## Determinism of CG's reduce API

CG API provides [Reduce() function](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#reduce) for performing a parallel reduction on the data provided by each thread in the specified group.

- However, in the programming guide, there is no mention of if this method is deterministic

- Another interesting bit

```
Hardware acceleration is used for reductions when available (requires Compute Capability 8.0 (Ampere Arch?) or greater).
A software fallback is available for older hardware where hardware acceleration is not available. Only 4B types are accelerated by hardware.
```



### Deterministic float-point reduction algorithm 

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


### Potential Hardware Implementation of Shuffle Command

Directly Register-to-Register Transfer: The shuffle instruction *`(SASS SHFL)`* utilizes a dedicated data path within the Streaming Multiprocessor (SM). It treats the registers
of all 32 threads in a warp as a shared pool for that single operation

[Warp-level Synchrony](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/): Because threads in a warp execute in lockstep on the hardware (SIMT), the data produced by one thread's register is available to be read by another thread's register in the same clock cycle or a fixed latency pipeline

Source Lane Selection: The hardware uses *`"source lane ID"`* to determine which register to read from

The performance implication is, this hardware feature was first introduced with the *Kepler Architecture*, 

the CUDA *`__shfl`* instructions are widely understood to be implemented on NVIDIA GPUs via a dedicated *`crossbar network "xbar"`* that connects the register files of the individual cores within a SM



```
No memory is needed apart from the SMX registers that hold the data before and after the instruction.

However there is no need to route the exchange via additional registers or memory (unlike previous compute capabilities, where shared memory was the only to move data between threads within the SM)

If my assumption is true that memory crossbar is reused for the shuffle instruction, then additional circuitry is needed to route
register accesses via that crossbar. Otherwise an crossbar would be needed
```

According to [one commenter](https://forums.developer.nvidia.com/t/shuffle-instructions-on-kepler-how-implemented/28504/3)

## Synchronization

## Appendix


### AtomicAdd


## Warp-Level Primitives

## shfl_

The parameter var

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
```

The parameter `var` represents the quantity (yes, it is pass-by-value) that the thread will make available for other warp-lanes to read. 




## References

[Cooperative Groups Flexible Groups of Threads](https://juser.fz-juelich.de/record/915940/files/08-aherten-cooperative-groups.pdf)

[Hardware vs. Software Implementation of Warp-Level Features in Vortex RISC-V GPU](https://arxiv.org/pdf/2505.03102)

[Kepler GK110/GK210 Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)

[Shuffle instructions on Kepler: How (is it) implemented](https://forums.developer.nvidia.com/t/shuffle-instructions-on-kepler-how-implemented/28504)

[How is shfl_sync implemented](https://forums.developer.nvidia.com/t/how-is-shfl-sync-implemented/216367)

More on general hardware concept

<!-- Great article on many concepts in computer system -->

[Lessons learned while building crossbar interconnects](https://zipcpu.com/blog/2019/07/17/crossbar.html)