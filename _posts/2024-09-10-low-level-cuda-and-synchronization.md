---
layout: post
title: "Low level CUDA and synchronization"
categories: low-level-GPU
---
```c
__threadfence(); // Ensure any pending memory operations are completed
asm("trap;");  // kill kernel with error
```

`asm("trap");` is an inline assembly instruction that triggers an immediate kernel abort. It's typically used to [stop kernel execution](https://stackoverflow.com/questions/34989481/how-to-interrupt-or-cancel-a-cuda-kernel-from-host-code) when an unrecoverable error condition is detected.

`assert()` function is also available for terminating a kernel

One use case is using this to active stop kernel running for debugging purpose. It can act like a breakpoint together with `cuda-gdb`.

Reading a register value

```c
__device__ unsigned int debug_output[1];

__global__ void myKernel() {
    // ...
    asm volatile("move.u32 %0, %%r0;" : "=r" (debug_output[0]));
    // ...
}
```

__What is the difference between `asm volatile("bar.sync 0;")` and `__threadfence()` in CUDA?__

1. Scope of synchronization:

  - `bar.sync 0` synchronizes all threads within a thread block
  - `__threadfence()` ensures memory operations are visible to other threads across the entire GPU

2. Type of synchronization:

  - `bar.sync` is a barrier synchronization that blocks threads until all reach that point
  - `__threadfence()` is a memory fence that ensures ordering of memory operations, but does not block threads

3. Level of control:

  - `bar.sync` allows specifying which named barriers to use (0-15) and how many threads should synchronize
  - `__threadfence()` is a general memory fence

4. Performance:

  - `bar sync` is typically faster as it only synchronizes within a block
  - `__threadfence()` has more overhead as it affects global memory across the entire GPU

`bar.sync` provides fine-grained control for synchronizing threads within a block, while `__threadfence()` ensures global memory consistency across entire GPU.

[CUDA: how to use barrier.sync](https://stackoverflow.com/questions/53662484/cuda-how-to-use-barrier-sync)

__block to block communication__

- A CTA (Cooperative Thread Array) is essentially equivalent to a thread block in CUDA terminology
- Multiple blocks/CTAs can run concurrently on a single SM (Streaming Mulitiprocessor)

Before Hopper

Different blocks can only reliablely communicate using global memory

Blocks cannot directly access each other's shared memory or L1 cache.

On newer GPU arch like Hopper, blocks in the same Cluster can use distributed shared memory for communication

["To mitigate the cost of transferring data across the gigantic die, H100 has a feature called `Distributed Shared Memory (DSMEM)`. Using this feature, applications can keep data within a GPC, or a cluster of SMs. This should allow for lower latency data sharing than the global atomics, while being able to share dat across more threads than would fit in a workgroup."](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)

DSMEM is implemented through thread block clusters, which allow a thread block to access the shared memory of other thread blocks within its cluster

[At the CUDA level, all DSMEM segments from thread blocks in a cluster are mapped into the generic address space of each thread, allowing direct referencing with simple pointers](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

DSMEM transfers can be expressed as async copy operations synchronized with shared memory-based barriers for tracking completion

```c
cluster.map_shared_rank(SMEM, DST_BLOCK_RANK)
```

[to get the shared memory address of a target block](https://arxiv.org/abs/2402.13499)

__\__threadfence() vs __syncthreads()__

1. Scope of sycnrhonization
   - `__threadfence()` ensures memory operations are visible across the entire GPU
   - `__syncthreads()` synchronizes threads only within a single thread block
  
2. Type of sychronization

   - `__threadfence()` is a memory fence that ensure ordering of memory operations, but does not block threads
   - `__syncthreads()` is a barrier synchronization that blocks threads until all reach that point
3. Memory consistency
   - `__threadfence()` enforces a memory ordering to ensure consistency across threads
   - `__syncthreads()` ensures all shared and global memory accesses made by threads in a block are visible to all other threads in that block

4. Usage
   - `__threadfence()` is typically used to coordinate memory accesses between different blocks.
   - `__syncthreads()` is used to coordinate memory accesses within a single block
  
5. Performance:
   - `__threadfence()` generally has more overhead as it affects global memory across GPU
   - `__syncthreads()` is typically faster as it only sychronize within a block

6. Blocking behavior
  - `__threadfence()` does not halt execution until data is written back to global memory
  - `__syncthreads()` blocks execution of all threads in a block until they all reach the synchronization point
  
`__threadfence()` is used for ensuring memory ordering across the entire GPU, while `__syncthreads()` is used for synchronizing threads within a single block.


```c
__global__ void combinedExample(int *input, input *output, int n)
{
  __shared__ int sharedSum;
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes a partial sum
  int partialSum = 0;
  for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
    partialSum += input[i];
  }

  // Reduce within the block
  atomicAdd(&sharedSum, partialSum);
  __syncthreads(); //Ensure all threads have contributed

  if (tid == 0) {
    // Block leader adds to global sum
    atomicAdd(output, sharedSum);
    __threadfence(); // Ensure global sum is visible to all blocks
  }
}
```

Without the block leader threadfence operation, we are risking a race condition for globalSum
