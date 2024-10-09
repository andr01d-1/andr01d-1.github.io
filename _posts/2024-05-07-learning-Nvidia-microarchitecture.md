---
layout: post
title: "Learning Nvidia GPU from Microarchitecture perspectives"
categories: computer-arch
---

<!-- Lower to the microarchitecture level -->

<!-- ```cuda

void VectorAdd()
{

}

``` -->

This is the first entry in a series of notes attempting to approach the study of GPU programming and computer architecture from a trajectory different
than the traditional one.

Let's jump right into the deep end from the beginning.

## Learning to use perf tools early

Starting with checking SM [Compute Capability(CC)](https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549?permalink_comment_id=5043495#gistcomment-5043495)

[Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/) is a great series from NVIDIA.

> global load efficiency `gld_efficiency` and global store efficiency `gst_efficiency` metrics that we might have used to ascertain whether our kernel does a good job of 
coalesced loads and stores.

If you are using the latest (as of 2024) Nsight version, the command has been updated.

```bash
ncu --devices 0 --query-metrics > my_metrics.txt
```

Interestingly, by default, my machine targets the `SM_52` architecture without specific indications.

```bash
nvcc mat_mat.cu
```

Download the complete code: <a href="/codes/mat_mat.cu" target="_blank">mat_mat.cu</a>

From aforementioned query, our card's CC is `SM_75`. Due to user [GPU permission issue](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters),

I had to rebuild the initrd after writing a configuration file to `/etc/modprobe.d`

```bash
sudo update-initramfs -u -k all
```

If it still doesn't work after reboot, one might have to switch to root user mode 

```bash
/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --call-stack --set detailed -k matrix_add_2D -o ncu_out_4 ./a.out
```

```bash
nsys profile -o mat_mat.nsysprofout --stats=true ./a.out
```

Command Line Interface (CLI) launch with specific metrics

```bash
/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sm,
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./a.out
```

```cpp
__global__ void matrix_add_2D(const arr_t *__restrict__ A,
                             const arr_t *__restrict__ B,
                             arr_t *__restrict__ C, const size_t sw,
                             const size_t sh) {


 size_t idx = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
 size_t idy = threadIdx.y + blockDim.y * (size_t)blockIdx.y;


 if ((idx < sh) && (idy < sw))
   C[idx][idy] = A[idx][idy] + B[idx][idy];
}
```

```bash
==PROF== Profiling "matrix_add_2D" - 0: 0%....50%....100% - 4 passes
Success!
==PROF== Disconnected from process 189133
[189133] a.out@127.0.0.1
 matrix_add_2D(const unsigned int (*)[1024], const unsigned int (*)[1024], unsigned int (*)[1024], unsigned long, unsigned long) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 7.5
   Section: Command line profiler metrics
   ----------------------------------------------- ----------- ------------
   Metric Name                                     Metric Unit Metric Value
   ----------------------------------------------- ----------- ------------
   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     request       65,536
   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector    2,097,152
   ----------------------------------------------- ----------- ------------
```

<img src="/images/memory_chart.png"/>


After applying <a href="/codes/mat_mat_v2.cu" target="_blank">the coalescing memory access fix</a>
by swapping `idx` and `idy` in previous version

```cpp
__global__ void matrix_add_2D(const arr_t *__restrict__ A,
                             const arr_t *__restrict__ B,
                             arr_t *__restrict__ C, const size_t sw,
                             const size_t sh) {


 size_t idx = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
 size_t idy = threadIdx.y + blockDim.y * (size_t)blockIdx.y;


 if ((idy < sh) && (idx < sw))
   C[idy][idx] = A[idy][idx] + B[idy][idx];
}
```


```bash
==PROF== Profiling "matrix_add_2D" - 0: 0%....50%....100% - 4 passes
Success!
==PROF== Disconnected from process 193209
[193209] a.out@127.0.0.1
 matrix_add_2D(const unsigned int (*)[1024], const unsigned int (*)[1024], unsigned int (*)[1024], unsigned long, unsigned long) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 7.5
   Section: Command line profiler metrics
   ----------------------------------------------- ----------- ------------
   Metric Name                                     Metric Unit Metric Value
   ----------------------------------------------- ----------- ------------
   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     request       65,536
   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector      262,144
   ----------------------------------------------- ----------- ------------
```


L1/Tex Cache hit is completely gone and L2 Cache hit rate is actually lower

<img src="/images/memory_chart_v2.png"/>



CUDA `__restrict__` has similar semantics to the corresponding one in C99, where it is also used to indicate that pointers are not aliased, [allowing more aggressive optimizations by the compiler](https://stackoverflow.com/questions/43235899/cuda-restrict-tag-usage).


If a kernel function parameter is a pointer to read-only data and is marked with both `__restrict__` and `__const__`, this may hint to the compiler that loads from this pointer
can be cached in a special read-only cache, potential improving performance with faster memory access and reduced memory traffic.

[Hardware Model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#hardware-model) section of NSight's doc has more in-depth explanation of metrics we would obtain from profiling. 

You might have heard the term `CTA` from CUDA Ninjas. 

> A (Thread) Block is array of threads, also known as a Cooperative Thread Array (CTA)
> The architecture can exploit this locality by providing fast shared memory and barriers between the threads within a single CTA.


CGA (Cooperative Grid Array)

> A new cooperative grouping introduced in the Hopper Architecture (SM90)
> [Clusters enable multiple thread blocks running concurrently across multiple SMs to synchronize and collaboratively fetch and exchange data](https://stackoverflow.com/questions/78510678/whats-cga-in-cuda-programming-model)


## Roofline Model Analysis

> The [Roofline performance model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/) provides an intuitive and insightful way to understand applicatoin performance, identify bottlenecks and perform optimization for HPC applications 

- [NERSC Roofline Model on NVIDIA GPUs](https://gitlab.com/NERSC/roofline-on-nvidia-gpus)

In the kernel examples

```cpp
__global__ void kernel_A(double* A, int N, int M)
{
  double d = 0.0;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) {
#pragma unroll(100)
    for (int j = 0; j < M; ++j)
    {
      d += A[idx];
    }

    A[idx] = d;
  }
}

```

As we can see

```assembly
$L__BB1_4:
        .pragma "nounruoll";
        add.f64         %fd13, %fd116, %fd1;
        ...
```
in a generated ptx segment <a href="/codes/unroll_pragma.ptx" target="_blank">unroll_pragma.ptx</a>

By using 

```bash
nvcc --ptx kernel_abc.cu
```

`pragma_unroll` is usually used for reducing loop overhead associated with loop control instructions and branch mispredictions in exchange of larger generated code size register pressure and etc.

Here is an [explanation](https://youtu.be/fsC3QeZHM1U?t=1352) of how to observe compute bound vs memory bound kernel behaviors in Nsight visualization

## Microarchitecture

> Arithmetic and other instructions are executed by the SMs; data and code are accessed from DRAM via the L2 cache. 

[GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

> The instructions are stored in global memory ([implemented with DRAM](http://homepages.math.uic.edu/~jan/mcs572/memory_coalescing.pdf)) that is inaccessible to the user but are prefected into an instruction cache during execution

[Where does CUDA Kernel Code reside on NVidia GPU](https://stackoverflow.com/questions/5121709/where-does-cuda-kernel-code-reside-on-nvidia-gpu)

The instruction cache is a separate memory space (for safety?) that is used to store the instructions being executed by the GPU's streaming multiprocessors (SMs)

[How CUDA Programming works](https://www.youtube.com/watch?v=QQceTDjA4f4) starts with introducing RAM attributes.

| Kepler Architecture            | The L1 instruction cache size for the GT100 (a Kepler-based GPU) is [reported to be 4KB](https://forums.developer.nvidia.com/t/instruction-cache/25521).
| Maxwell Architecture           | For Maxwell architecture, specifically the sm_5x series, the instruction cache size is also mentioned as [4KB](https://forums.developer.nvidia.com/t/code-instruction-cache/38939).
| Pascal Architecture            | The L1 instruction cache size for Pascal is mentioned as [8KB](https://forums.developer.nvidia.com/t/instruction-cache-and-instruction-fetch-stalls/76883).
| Volta and Ampere Architectures | For the Volta and Ampere architectures, the instruction cache size is not explicitly detailed in the provided sources, but the combined L1 data and shared memory cache sizes are 128 KB per SM for Volta and 192 KB per SM for [Ampere](https://forums.developer.nvidia.com/t/instruction-cache-size-for-ampere-and-volta-arch/251422) 

A widely cited technical report is [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826)

Both the host and the device can access 'global memory'; this interaction is mediated through the CUDA runtime, which handles memory allocation, data transfer, and synchronization to ensure data coherence."

"Memory mapping" operations do imply PCIe operations. for example, [BAR (Base Address Registers)](https://stackoverflow.com/questions/30190050/what-is-the-base-address-register-bar-in-pcie)

Due to latency and bandwidth characteristics of PCIe, [performance considerations](https://superuser.com/questions/1545812/can-the-gpu-use-the-main-computer-ram-as-an-extension) are crucial

Many specially designed SoCs, such as Apple's M series processors, have a different memory model. Perhaps this topic is for another note.

Further reading: [**Memory Latency Benchmarking**](https://en.algorithmica.org/hpc/cpu-cache/pointers/)


### [Memory Hierarchy](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and Operations

**local memory** or thread-local (global) memory is actually [global memory](https://stackoverflow.com/questions/10297067/in-a-cuda-kernel-how-do-i-store-an-array-in-local-thread-memory). 

#### Constant Cache

is a specialized read-only cache used in GPUs to store constant data that is shared across multiple threads. This type of cache is particularly efficient when all threads across the same memory address, as it can broadcast a single value to all threads in a warp, minimizing memory access latency. 

Per core L1 cache access is generally considered to be [very fast](https://stackoverflow.com/questions/30371708/what-are-the-access-times-for-different-gpu-memory-spaces) [(low number of cycles)](https://forums.developer.nvidia.com/t/fermi-l2-cache-how-fast-is-the-l2-cache-how-do-i-access-it/23176/8) according to numbers obtained on Fermi and [Kepler](https://stackoverflow.com/questions/4097635/how-many-memory-latency-cycles-per-memory-access-type-in-opencl-cuda).


The content can be modified from the host side using the `cudaMemcpyToSymbol` function. This allows for flexibility in updating constant values between [kernel launches](https://carpentries-incubator.github.io/lesson-gpu-programming/constant_memory.html).

#### Texture Cache

Texture units have their own [L1 texture caches](https://stackoverflow.com/questions/76383420/ampere-constant-memory-and-read-only-cache), and the entire GPU [shares](https://computergraphics.stackexchange.com/questions/355/how-does-texture-cache-work-considering-multiple-shader-units) a [single L2 cache](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).

It is [reported](https://chipsandcheese.com/2021/04/16/measuring-gpu-memory-latency/) that
> Nvidia sticks with a more conventional GPU memory subsystem with only two levels of cache and high L2 latency


<p align="center">
    <img src="https://www.techspot.com/articles-info/2729/images/2023-09-01-image-9.jpg" />
</p>

[From Why GPUs are the new Kings of Cache. Explained](https://www.techspot.com/article/2729-cpu-vs-gpu-cache/)

#### Instruction Cycles

> On Volta, registers are divided into two
64-bit wide banks. One Volta instruction can only access 64 bits of each bank
per clock cycle. Thus an instruction like FFMA (single precision floating-point
fused multiply-add operation) can read at most two values from each bank
per clock


#### Register Bank Conflicts and Register Reuse Cache

[Register Cache](https://developer.nvidia.com/blog/register-cache-warp-cuda/)

<p align="center">
    <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/paper4-625x348.jpg" />
</p>

<p style="text-align: center;"><a href="https://developer.nvidia.com/blog/register-cache-warp-cuda/">Execution and Memory hierarchy in CUDA GPUs.</a></p>

Scott Gray has written a [comprehensive article](https://github.com/NervanaSystems/maxas/wiki/SGEMM#calculating-c-register-banks-and-reuse) on Register Banks and Reuse that provides valuable insights. 

<!-- TODO: Add VectorAdd for instruction -->

The compiler outputs a string of control variables, which control the sequence of micro-operations (uops), known as a control word. The micro-operations specified in a control word are called microinstructions.

Higher-level language constructs such as `if`, `for`, and `while` [compile directly](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-34-gpu-flow-control-idioms) into GPU assembly instructions, to avoid [hazards](https://en.wikipedia.org/wiki/Hazard_(computer_architecture))


Similar to aforementioned coalescing example,
> `shared_ld_bank_conflict`
> `shared_st_bank_conflict`
> `shared_efficiency`
> `shared_load_transactions_per_request`

can be used to check `bank conflicts` in a kernel.

References: [GPU Hardware Effects](https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md)

<!-- Turns out even SASS code might not be the "real instructions" -->

**Reuse Flags**

> "Why reuse (register) mitigates register bank conflicts"
"The least significant bit in reuse flags controls the cache for the first source operand slot"
"The most significant bit is for the fourth source operand slot"

**Wait barrier mask; read/write barrier index**

While most instructions have fixed latency and can be statically scheduled by the assembler, instructions involving memory and shared resources typically have variable latency. 

<!-- will stall the later instruction until the results of the earlier one are  -->


**Instruction Encoding**

Early GPU architectures (e.g. Fermi) is reported to use SIMD lane [masking](https://citeseerx.ist.psu.edu/document?doi=afda3825b7419c55f31d8d2f487206b263063ef3&repid=rep1&type=pdf) and 

**Instruction decoding**

GPUs typically have fewer instruction decoders compared to the number of cores, and a group of cores may only execute one or two different code paths at any given time. This suggests a more [streamlined approach](https://cs.stackexchange.com/questions/121080/what-are-gpus-bad-at) to instruction decoding and execution rather than a traditional stack-based approach. 


<!-- LSU (Load Store Unit)

Maybe it is time to take a look at cuAssembler?
[CuAssembler Author introduction](https://www.zhihu.com/people/xiaoguiren/posts)

Reverse compile of nvidiasm -->

When it comes to understanding the [overheads of launching CUDA Kernels](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf), they are often [cited as being 5 microseconds or involving a few thousands of instructions](https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413/8)

*Pinned memory* allows for *faster* data transfer between the host and the device becaues it eliminates the need for an intermediate copy to a staging area. This is achieved through DMA. 

[Interleaving streams streams help hide latencies](https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod14DataXfer/Mod14DataXfer.pdf)

#### Shared memory resource

> Shared memory is shared amongst all warps of a single block. The amount of memory that a block requires is either inferred by the compiler from variable declarations (`__shared__ int sharedArray[1024]`) or it can be explicitly passed as a parameter when launching a kernel (`kernel<<gridDim, blockDim, sharedMemorySize, stream>>(...)`);

Perf metric to observe: `achieved_occupancy`


#### Cooperative Thread Array (block)

Cooperative Groups is introduced in CUDA 9, which aims to satisfy `safety`, `maintainability` and `modularity` by making [synchronizatoin an explicit part of programming model](https://developer.nvidia.com/blog/cooperative-groups/)
 
> one SM can run several concurrent CUDA blocks depending on the resources needed by CUDA blocks. Each kernel is executed on one device and CUDA supports running multiple kernels on
> a device at one time.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2020/06/kernel-execution-on-gpu-1.png"/>

[CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

