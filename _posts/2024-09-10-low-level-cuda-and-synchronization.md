---
layout: post
title: "Low Level CUDA and Synchronization"
categories: low-level-GPU
---
```c
__threadfence(); // Ensure any pending memory operations are completed
asm("trap;");  // kill kernel with error
```

`asm("trap");` is an inline assembly instruction that triggers an immediate kernel abort. It's typically used to [stop kernel execution](https://stackoverflow.com/questions/34989481/how-to-interrupt-or-cancel-a-cuda-kernel-from-host-code) when an unrecoverable error condition is detected.

`assert()` function is also available for terminating a kernel

One use case is using this to actively stop kernel running for debugging purpose. It can act like a breakpoint together with `cuda-gdb`.

Reading a register value

```c
__device__ unsigned int debug_output[1];

__global__ void myKernel() {
    // ...
    asm volatile("move.u32 %0, %%r0;" : "=r" (debug_output[0]));
    // ...
}
```

## __threadfence()

`__threadfence()` ensures global memory consistency across entire GPU. As a (memory fence) operation, it halts the execution of subsequent *memory* operations have completed and are "visible". Unlike `__syncthread()`, it is not a full thread halt. 

<!-- It specifically stalls the memory operation pipeline -->

### PTX

`MEMBAR.SC.GPU` acts as a [strong memory barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations) that enforce Sequential Consistency (SC) for all threads on the GPU. ensuring loads/stores happens in the order intended by the programmer, crucial for synchronization when threads share data.

<!-- TODO: PTX's memory model and scope -->

### [Why do we need `__threadfence()`](https://forums.developer.nvidia.com/t/is-threadfence-useful-at-all/244413)

CUDA follows a [relaxed memory consistency model](https://en.wikipedia.org/wiki/Consistency_model#Relaxed_memory_consistency_models). This means the order in which memory operations (reads and writes) appear to occur may not match the order in they were written in your code, unless the explicit synchronization is used.

<!-- Changes made by one thread are not immediately guaranteed -->

CUDA does not guarantee global memory write [visibility](https://forums.developer.nvidia.com/t/best-way-to-communicate-small-amount-of-data-across-ctas/222852) across threads within an iteration.

CUDA 12.x and newer emphasize "scoped" operations. Consistency is often defined by [a scope (e.g., thread, block, device, or system)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html), allowing you to restrict synchronization to only the necessary groups of threads for better performance


<details>
  <summary>a side note on block to block communication</summary>

  <p> A CTA (Cooperative Thread Array) is essentially equivalent to a thread block in CUDA terminology </p>
  <p> Multiple blocks/CTAs can run concurrently on a single SM (Streaming Multiprocessor) </p>
</details>

Before Hopper

Different blocks can only reliably communicate using global memory

Blocks cannot directly access each other's shared memory or L1 cache.

On newer GPU arch like Hopper, blocks in the same Cluster can use distributed shared memory for communication

["To mitigate the cost of transferring data across the gigantic die, H100 has a feature called `Distributed Shared Memory (DSMEM)`. Using this feature, applications can keep data within a GPC, or a cluster of SMs. This should allow for lower latency data sharing than the global atomics, while being able to share dat across more threads than would fit in a workgroup."](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)

<details>
  <summary>Number of SMs per GPC (Graphics Processing Cluster)</summary>

  varies by architecture

  <p>8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU <a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">Hopper</a></p>

  <p>8 GPCs, 8 TPCs/GPC, 2 SMs/TPC, 16 SMs/GPC, 128 SMs per full GPU <a href="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/">Ampere</a></p>

</details>

DSMEM is implemented through thread block clusters, which allow a thread block to access the shared memory of other thread blocks within its cluster

[At the CUDA level, all DSMEM segments from thread blocks in a cluster are mapped into the generic address space of each thread, allowing direct referencing with simple pointers](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

DSMEM transfers can be expressed as async copy operations synchronized with shared memory-based barriers for tracking completion

```c
cluster.map_shared_rank(SMEM, DST_BLOCK_RANK)
```

[to get the shared memory address of a target block](https://arxiv.org/abs/2402.13499)

## __syncthreads()

is used to ensure all shared and global memory accesses made by threads in a block are visible to all other threads in that block.

*__syncthread()* is about *execution flow* and *memory visibility*, while `__threadfence()` is solely about *memory visibility* and *ordering*, allowing threads to continue progressing.


### PTX

`bar.sync` (and its counterpart `__syncthreads()`) ensures that all threads within a CTA or a block reach that point before any threads proceeds. Starting with VOlta, barriers are enforced per thread and will not succeed until reached by all non-existed threads in the block.

It can not be used to synchronize threads across different SMs or across different thread blocks. It is localized to the multiprocessor running that block. 

`bar.sync` allows specifying which named barriers to use (0 - 15) and how many threads should synchronize


It is possible to synchronize subsets of threads within a block (e.g., partial warps). The 16 logical barrier resources (or "named barriers") are hardware-level counters.

`ID 0` is typically reserved for standard synchronization. `__syncthreads()` often maps to `bar.sync 0`, which synchronizes all threads within a thread block

All threads participating in a specific synchronization point must use the same ID. One thread in a group uses ID 1 and another uses ID 2 to wait for each other, they will deadlock.

<!-- `bar.sync` provides fine-grained control for synchronizing threads within a block, while `bar.arrive` -->

`bar.arrive` increments the counter while `bar.sync` waits and once the count is reached, resets the counter for future uses.


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

Without the block leader __threadfence operation, we are risking a race condition for globalSum

## What about __syncwarp()

is a CUDA synchronization primitive used to synchronize threads within a warp

It is typically used when threads in a warp need to perform more complicated communications or collective operations than what data exchange primitive provide. It is particularly important for ensuring correct behavior in divergent code paths within a warp.

It was introduced in CUDA 9 to address challenge arising from [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling).

It is potentially implemented with registers and has only one warp-level "barrier", allowing divergent threads to call it at different program counters and still converge.

**Performance Implication**

[Warp-level synchronization with `__syncwarp()` can be faster than global/shared reduction/atomic operation when there is high contention, as it doesn't use atomic hardware](https://accelsnow.com/CUDA-Warp-Primitives-and-Sync-Notes)

## References

[CUDA Memroy Model by CUDA Community Meetup Group](https://www.youtube.com/watch?v=VJ1QLrmfQws)

[GPU Memory Consistency: Specification, Testing, and Opportunities  for Performance Tooling](https://www.sigarch.org/gpu-memory-consistency-specifications-testing-and-opportunities-for-performance-tooling/)

[CUDA: how to use barrier.sync](https://stackoverflow.com/questions/53662484/cuda-how-to-use-barrier-sync)

[Difference between __syncthread() and __threadfence()](https://forums.developer.nvidia.com/t/difference-between-syncthreads-and-threadfence/203817/2)