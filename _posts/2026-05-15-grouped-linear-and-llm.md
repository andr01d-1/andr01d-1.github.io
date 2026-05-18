---
layout: post
title: "Grouped Linear Top to Bottom"
categories: Sparse-tensor-core LLM Grouped-Linear
---

Linear layer, fully connected layer (FC) and affine transformation calculating

$$y = xA^T + b$$ which connects every input neuron to every output neuron. These concepts are all connected.

For a `Grouped Linear` layer, only a specific part of the inputs connects to a specific part of the outputs.

Computationally, group linear operations are closely related to block-sparse (or structured sparse). 

The concept comes from the CNN era, [grouped convolution](https://blog.yani.ai/filter-group-tutorial/), a grouped linear layer is a mathematically equivalent variation of a grouped 1x1 convolution (point-wise convolution), where the filter size is $1 \times 1 \times C$.

### Math Intuition

Grouped linear operations are related to blocks-sparse.


## Hardware Design for Structured Sparsity

N:M Find-Grained Sparsity is a semi-structured pruning technique. For every contiguous group of $M$ weights in a matrix, only $N$ are retained while the rest are zeroed out. Most commonly implemented as 2:4 sparsity (where 50% of weights are pruned)

[Nvidia's Sparse Tensor Cores exploit a 2:4 (50%) sparsity pattern that leads to twice the math throughput of dense matrix units](https://arxiv.org/pdf/2104.08378).


### Block Sparse Format

There are multiple SpMM (Sparse-matrix dense-matrix multiplication), software is responsible for translating [multiple compressed storage format](https://developer.nvidia.com/blog/establishing-a-scalable-sparse-ecosystem-with-the-universal-sparse-tensor/).

The storage formats (CSR, COO, BSR or etc.) are known only to the software layer. 

The hardware is fixed for "2:4 sparsity". For every four compressed values, we would need one extra byte for storing their original relative indices, as illustrated in below diagram.

<figure style="width: 100%; margin: 0 auto; text-align: center;">
    <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2021/06/sparse-tensor-cores.png" style="width: 100%">
    <figcaption>GEMM using block sparse weights and dense activations. </figcaption>
</figure>



### Hardware 


$$\mathbf{C} = \alpha \cdot \operatorname{op}(\mathbf{A}) \cdot \operatorname{op}(\mathbf{B}) + \beta \cdot \mathbf{C}
$$

[In this operation, `A` is a sparse matrix of size `MxK`, while `B` and `C` are dense matrices of size `KxN`, `MxN`, respectively.](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)

<figure style="width: 100%; margin: 0 auto; text-align: center;">
    <img src="https://www.glennklockwood.com/garden/attachments/NVIDIA-A100-Structured-Sparsity.png" style="width: 100%">
    <!-- <figcaption>GEMM using block sparse weights and dense activations. </figcaption> -->
</figure>

I haven't found any description of hardware implementation details, the illustration suggests, `indices` serve as control inputs for the [multiplexer](https://en.wikipedia.org/wiki/Multiplexer) select lines. This ensures that only the specific dense matrix (input activation) element indexed is forwarded to the multiplier input, effectively bypassing zero values.

TODO:

Finish reading the paper

### Using Sparse Tensor Core

The TensorRT or cuSPARSELt pipeline are the common path for using the NVIDIA GPU sparse tensor core

Since the sparsity is built in Tensor Core hardware. It is more reasonable to use CUTLASS


```cpp

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"

using namespace cute;

// 1. Define Data Types
using ElementA        = cutlass::half_t;     // Sparse Matrix A (Pre-compressed weights)
using ElementB        = cutlass::half_t;     // Dense Matrix B (Activations)
using ElementC        = cutlass::half_t;     // Output Matrix C
using ElementAccum    = float;               // Precision inside Tensor Cores (FP32)

// 2. Define Matrix Layouts (Row-major vs Column-major)
using LayoutA         = cutlass::layout::RowMajor;
using LayoutB         = cutlass::layout::ColumnMajor;
using LayoutC         = cutlass::layout::RowMajor;

// 3. Define the Target Hardware Core and Shape
// For Ampere/Hopper Tensor Cores with 2:4 Sparsity
using OpClass         = cutlass::arch::OpClassTensorOp;
using ArchTag         = cutlass::arch::Sm80; // Sm80 for Ampere, Sm90 for Hopper

// Threadblock tile sizes (M, N, K)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>; // The underlying m16n8k32 math unit shape

// 4. Define the Epilogue (Activation function applied after MatMul)
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccum, ElementAccum>;

// 5. Tie it all together using the Structured Sparse Gemm template
using SparseGemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithSparsityConfiguration,
    2, // Number of pipeline stages for double buffering
    cutlass::arch::SparseMode::kSparse24 // <-- Enforces the 2:4 Hardware Sparsity configuration
>::GemmKernel;

// Instantiate the Universal Device Driver Wrap
using Gemm = cutlass::gemm::device::GemmUniversalBase<SparseGemmKernel>;
```

Pipeline

```cpp
// Allocate your host-side configuration
typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, // Standard GEMM Execution mode
    {M, N, K},                               // Problem size dimensions
    1,                                       // Batch count
    {alpha, beta},                           // Scaling factors
    d_compressed_A,                          // Pointer to pre-compressed weight matrix
    d_B,                                     // Pointer to dense input activation matrix
    d_C,                                     // Pointer to output matrix source
    d_D,                                     // Pointer to final destination output matrix
    d_metadata,                              // Pointer to your pre-generated 2-bit metadata matrix
    // ... Layout strides and workspace configurations go here
};

// Initialize the CUTLASS object
Gemm gemm_op;
cutlass::Status status = gemm_op.initialize(arguments, workspace_ptr, stream);

// Execute the hardware-sparse kernel on your CUDA stream
if (status == cutlass::Status::kSuccess) {
    status = gemm_op(stream);
}
```

A SM-80 FP16 example

```cpp

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUTLASS core and GEMM configuration headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"

// Helper macros for CUDA error checking
#define CUDA_CHECK(status)                                              \
    if (status != cudaSuccess) {                                        \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status)      \
                  << " at line " << __LINE__ << std::endl;              \
        return -1;                                                      \
    }

#define CUTLASS_CHECK(status)                                           \
    if (status != cutlass::Status::kSuccess) {                          \
        std::cerr << "CUTLASS Error: " << cutlassGetStatusString(status)\
                  << " at line " << __LINE__ << std::endl;              \
        return -1;                                                      \
    }

int main() {
    // 1. Define Matrix Dimensions (M, N, K)
    // For 2:4 sparsity, the inner K-dimension must be a multiple of 32
    const int M = 512;
    const int N = 256;
    const int K = 512; 

    // 2. Define CUTLASS Template Configurations
    using ElementA        = cutlass::half_t;     // Sparse Weights Matrix A (Pre-compressed)
    using ElementB        = cutlass::half_t;     // Dense Inputs Matrix B
    using ElementC        = cutlass::half_t;     // Output Matrix C
    using ElementAccum    = float;               // Math Core Accumulator Precision (FP32)

    using LayoutA         = cutlass::layout::RowMajor;
    using LayoutB         = cutlass::layout::ColumnMajor;
    using LayoutC         = cutlass::layout::RowMajor;

    using OpClass         = cutlass::arch::OpClassTensorOp;
    using ArchTag         = cutlass::arch::Sm80; // Targeting Ampere (A100, RTX 30/40, etc.)

    // Tile shapes matching the physical Sparse Tensor Core hardware mapping
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>; // Underlying m16n8k32 sparse math unit

    // Linear combination Epilogue: Computes D = alpha * (A x B) + beta * C
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccum, ElementAccum>;

    // 3. Assemble the Structured Sparse Configuration Template
    using SparseGemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithSparsityConfiguration,
        2, // Number of pipeline stages for memory masking optimization
        cutlass::arch::SparseMode::kSparse24 // <--- Triggers 2:4 Hardware Sparsity Execution
    >::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalBase<SparseGemmKernel>;

    // 4. Calculate Allocation Sizes
    // Due to 2:4 structured compression, Matrix A requires half the storage space of dense K
    size_t size_A_compressed = M * (K / 2) * sizeof(ElementA);
    size_t size_B            = K * N * sizeof(ElementB);
    size_t size_C            = M * N * sizeof(ElementC);
    
    // 2-bit metadata tracking structure requirements: 
    // 4 elements are packed into 2 bits, meaning Metadata size is exactly 1/8th of the uncompressed K shape
    size_t size_metadata     = M * (K / 8) * sizeof(uint8_t);

    // 5. Allocate Device Buffers on the GPU
    void *d_A_compressed, *d_B, *d_C, *d_D, *d_metadata;
    CUDA_CHECK(cudaMalloc(&d_A_compressed, size_A_compressed));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_D, size_C)); // Destination matrix shares output footprint
    CUDA_CHECK(cudaMalloc(&d_metadata, size_metadata));

    // Note: In a production inference engine, you would load your offline-pruned 
    // and compressed weights/metadata buffers directly into these pointers via cudaMemcpy.

    // 6. Setup Execution Arguments and Pipeline Factors
    float alpha = 1.0f;
    float beta  = 0.0f; // Pure multiplication without bias addition

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, // Universal matrix execution state
        {M, N, K},                               // Dense target tensor problem size
        1,                                       // Batch execution count
        {alpha, beta},                           // Element-wise scale factors
        d_A_compressed,                          // Pointer to pre-compressed 2:4 data values
        d_B,                                     // Pointer to live dense activation values
        d_C,                                     // Input source reference pointer
        d_D,                                     // Target calculation destination pointer
        d_metadata,                              // Pointer to the pre-packaged 2-bit placement map
        M,                                       // Strides for row tracking transformations
        K / 2,                                   
        K,
        M,
        M
    };

    // 7. Query Workspace Buffer Requirements
    Gemm gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    // 8. Initialize and Launch the Kernel on a CUDA Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cutlass::Status status = gemm_op.initialize(arguments, d_workspace, stream);
    CUTLASS_CHECK(status);

    // Dispatch the custom compiled kernel directly to the Sparse Tensor Cores
    status = gemm_op(stream);
    CUTLASS_CHECK(status);

    // Synchronize to ensure computation completes before extracting outputs
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Successfully executed 2:4 Structured Sparse GEMM via CUTLASS!" << std::endl;

    // 9. Clean up Resources
    cudaFree(d_A_compressed);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_metadata);
    if (d_workspace) cudaFree(d_workspace);
    cudaStreamDestroy(stream);

    return 0;
}
```


```bash
nvcc -std=c++17 -O3 \
     -arch=sm_80 \
     -I /path/to/cutlass/include \
     sparse_gemm_example.cu -o sparse_gemm_example

```

### Dummy Data

To actually run this example, we would need to generate valid dummy data. 

To quickly test the CUTLASS pipeline without setting up a full model pruning loop, we can populate the metadata buffer with a repeating pattern that the hardware recognizes as a valid 2:4 structure

The easiest valid pattern is choosing the first 2 elements out of every 4 (indices 0 and 1)

- in binary, picking indices 0 and 1 for a 4-element block is encoded as `0x4` (or `0100` in the hardware's specific configuration map).
- When we pack four of these 4-element blocks into a single 8-bit byte (`uint8_t`), the byte value becomes `0x44`

```cpp
// Fill the metadata buffer with 0x44 (binary pattern picking indices 0 and 1)
// This ensures the hardware maps the data lanes correctly and won't crash
cudaMemset(d_metadata, 0x44, size_metadata);
```

What would happen if the metadata is invalid?

- Silent Data Corruption: The GPU will run at full speed but pull the wrong activations from input, giving completely wrong output numbers.
- Hardware Exceptions: If the hardware reads an undefined bit combination that doesn't map to a valid 2:4 selection, it can trigger an execution fault inside the Tensor Core, causing the CUDA context to crash

back to the math formulation

$$y = xA^T + b$$

where 
- $x$ (Activation) is the input stored as `[Batch Size, Input Features]`
- $A$ (Weight): PyTorch Stored internally as `[Output Features, Input Features]` 
- $A^T$ (Transposed Weights): To make the inner dimensions match $x$, PyTorch transposes the weights to `[Input Features, Output Features]`
- $y$: `[Batch_Size, out_features]`

PyTorch defaults to the Channel First (NCHW) memory format. Early versions of cuDNN were originally optimized for such layout. For Convolution Weights (Output, Input, Height, Width). However with the introduction of TensorCore, [it is actually faster to use NHWC layout](https://stackoverflow.com/questions/44280335/how-much-faster-is-nchw-compared-to-nhwc-in-tensorflow-cudnn). (I wonder what the motivation of this change though?)

For LLMs, above formulation enables us to stack input sequences row-by-row.

### Gradient

According to multivariate chain rule (Backpropagation). The gradient for a specific weight $w$ in a layer is typically computed as the product of three partial derivatives:

$$\frac{\partial{L}}{\partial{a}} \times \frac{\partial{\alpha}}{\partial{z}} \times \frac{\partial{z}}{\partial{w}}$$

where 

- $\frac{\partial{L}}{\partial{a}}$ The derivative of the final loss function with respect to the neuron's output activation
- $\frac{\partial{\alpha}}{\partial{z}}$ The derivative of the activation function with respect to the pre-activation
- $\frac{\partial{z}}{\partial{w}}$ The derivative of the pre-activation $z$ (i.e., the weighted sum of inputs) with respect to the weight

where `pre-activation` is the linear transformation applied to previous layer's output.

- $z = \sum_{i=1}^{n}(w_i \cdot x_i) + b$ for a single neuron

For a standard pipeline

1. Linear step: $z = Wx + b$ (Pre-activation)
2. Normalize Step: $\hat{z} = BatchNorm(z)$
3. Non-linear Step: $a = \sigma(\hat{z})$ (Post-activation)

Two motivation for applying batch norm for controlling the Distribution: Pre-activations can easily grow too large or drift during training (Internal Covariate Shift). Normalizing $z$ forces the mean to $0$ and variance to $1$, ensuring the values land in the stable, active zones of your activation function.

#### Math Modification

Because Batch Normalization includes its own learnable shift parameter $\beta$, the classical bias term $b$ becomes redundant and is typically removed $(z = Wx)$.

This is functionally equivalent to fusing the bias directly into the normalization layer.


$$ \hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

$$ y = \gamma \hat{z} + \hat{\beta} $$

where $\gamma$ is scaling factor, $\mu$ is the mean, $\epsilon$ is a tiny constant added for numerical stability

Which is the reason we should set `bias=False` in any `Linear` or `Conv2d` layer that is immediately followed by a `BatchNorm` layer. If there is an activation function like `ReLU` between the weight layer and the BN layer, we may set `bias=True`.


#### Inference vs Training

During inference, `BatchNorm` simplifies to linear transformation.

$$ y = \gamma(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}) + \beta $$

After rearrange the terms to separate the input $(x)$:

$$ y = (\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}})x + (\beta - \frac{\gamma \cdot \mu}{\sqrt{\sigma^2 + \epsilon}}) $$



### Other Intuition

Introducing noise acts as an effective form of regularization. Quantization can act as a form of regularization. 

Microscaling (block based mxfp4) actually works like a form of quantization

## Grouped Linear and MoE

Traditionally, routing means tokens are scattered to many different experts. When computing the forward and backward passes, this leads to numerous, independent GEMMS in the compute stream. 

- Because different experts receives different numbers of tokens, standard batching fails 
- The overhead of launching separate GPU kernels for each expert leads to high latency and poor hardware utilization

The solution: Grouped GEMM / Grouped Linear

batches all expert calculations together despite varying token volumes. The model stores the weights of all experts allocated as a contiguous block/tensor in GPU memory. 



- Ragged Batching
- Grouped Linear



TODO:

DeepGEMM from deepseek and Transformer Engine?





## References

A paper from NVIDIA

[Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378)

[Accelerating Matrix Multiplication with Block Sparse Format and Nvidia Tensor Cores](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)

[Exploiting Ampere Structured sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)

[How to access sparse tensor core functionality in CUDA](https://stackoverflow.com/questions/74018900/how-to-access-sparse-tensor-core-functionality-in-cuda)

[A gentle introduction to GEMM using mma tensor cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/)

[Accelerating large scale mixture of experts training in Pytorch](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)

[NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

[The longest PTX instructions for Tensor Cores](https://ashvardanian.com/posts/longest-ptx-instruction/)

Structured sparsity is a type of [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))

[github NM-sparsity](https://github.com/aojunzz/NM-sparsity)