---
layout: post
title: "Micronote on Microscaling"
categories: VLSI Microscaling micro-architecture
---

## What is MicroScaling

[IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) is the most widely used technical standard for binary floating-point arithmetic. 

Standard precision like FP32 and FP64 has been used extensively in general computation and [High-performance computing](https://en.wikipedia.org/wiki/High-performance_computing) world.

With the raise of Deep Neural Networks, low bit data type becomes popular due to potential to drastically reduce model size, memory usage, power consumption and latency. 

[Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) is [further proposed](https://arxiv.org/pdf/2310.10537) to leverage "efficiency improvement" low bit format brought in a more "granular" fashion. 


The design trade off is 

## Hardware Efficiency vs Accuracy


"[Using $ INT8 \times INT8 $ with accumulation into fixed point (FX) is the cheapest](https://arxiv.org/pdf/2303.17951) and is dominated by the multiplication, whereas using floating point for either operand or accumulation formats are (often hugely) dominated by the cost of the accumulation ("alignadd" + "normacc") "

A potential implementation is using FP8 operands with a *fixed point* accumulator.

Historically, [fixed-point](https://en.wikipedia.org/wiki/Fixed-point_arithmetic) DSPs are used in a [great number of high volume applications](https://www.analog.com/en/resources/technical-articles/fixedpoint-vs-floatingpoint-dsp.html)

Energy usage, area of FP8 FMA could be 40 ~ 50 percent more than INT8 FMA. This is a large part of why most dedicated [ML inference chips use INT8](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/).

[Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) is a critical concept in Digital Signal Processing and Communication. Deep Learning has been heavily borrowing the concept to optimize models.

<figure style="width: 100%; margin: 0 auto; text-align: center;">
    <img src="https://www.ai-bites.net/content/images/2023/09/image-4.png" style="width: 100%;">
    <figcaption>Symmetric vs Asymmetric Quantization</figcaption>
</figure>

```
Symmetric Clipping. Floating-point formats are naturally symmetric around zero. In contrast,
signed integers in two’s complement have one extra negative value:
```

<!-- $ Q_{min}=-2b-1 $ and $ Q_{max}=2b-1-1 $ -->

```
We find that this asymmetric range usually does not affect inference. 
However, it degrades INT8 training due to a persistent negative bias in gradients.
```

***Symmetric Quantization is more hardware efficient***

By eliminating the zero-point, matrix multiplication becomes a pure integer multiplication and accumulation;
There is no need to store zero-point parameters, more memory is saved, and less bandwidth is required to load parameters.

***Layer Norm (LN) is generally considered more hardware-friendly (for Transformers)***

- No Batch Dependency: Calculates statistics per sample, avoiding dependency on batch size and cross-device communication in distributed training
- Consistent Inference: Works the same in training and inference, simplifying deployment
- GPU Parallelism: Its per-sample nature aligns well with GPU architecture

***Batch Norm (BN) is more popular in CNN***

- In training, dynamic mini-batch statistics (mean/variance) for stable updates
- In inference, fixed, averaged population statistics (running mean/variance) collected during training are retained and used
- Data across batches shares similar spatial pixels and channel statistics


In notation denoted in [paper](https://www.usenix.org/system/files/atc21-zhou.pdf), for parameters like "$S_w$, $S_x$, $S_y$..." Let's assume $W$ is symmetrically quantized while the input $x$ and output $y$ activations are asymmetric (very common scenarios).

a float point divider" would introduce multi-cycle latency overhead. Integer approximation is a common solution to trade accuracy for latency. Something like a scale factor that can be done through "bit shifting" which is extremely efficient in hardware implementation.

***Takeaway: use integer scale factors with higher precision to minimize post-MAC quantization loss.***

## Microscaling Strategies

#### Weight Quantization (Memory Reduction)

Usually static, directly reduces the storage size of the model

#### Activation Quantization (Speed & Efficiency)

Enables faster integer arithmetic on hardware and reduces memory bandwidth pressure, dynamic or static. Activations are dynamic (data dependent), making them harder to quantize than weights because their range changes with input data

#### Strategies

Per-tensor Scaling:

- Scale factor is computed for the entire weight and activation tensor/matrix
- Lowest overhead of hardware acceleration design
- Model accuracy deteriorates at 8-bit, especially when dynamic range is different across channels

Per-channel Scaling

- Computed for each activation channel and weight kernel
- Metadata overhead increase
- Lossless model accuracy at 8-bit for small and large models

Per-block Scaling (e.g. [VS-Quant](https://arxiv.org/pdf/2102.04503))
- For each activation and weight vector or micro-tensor
- Metadata overhead, hide with careful hardware design

##### Per-block scaling

With 1-level scaling
  - Example: microsoft [floating point MSFP](https://www.microsoft.com/en-us/research/blog/a-microsoft-custom-data-type-for-efficient-inference/)
  - Essentially block floating-point
  
Saving both memory and computation by [Shared Exponent](https://arxiv.org/abs/2310.10537). By identifying the maximum value, compute the exponent

<!-- TODO: Maximum value using a reduction tree (single cycle, combinational logic) -->

With 2-level scaling
- Example: [VS-Quant](https://arxiv.org/pdf/2102.04503)
- Again!! Design to balance accuracy and hardware efficiency

With 2-level hierarchical scaling

Example: MX format

Similar to VS-Quant, but level 0 and level 1 scaling are at stellar granularity of 2 and 16 respectively

Level 0 and level scale factors use E1M0 and E8M0 (8 bit unsigned) type respectively

**Block level mixed format** (Code book block)
- Choose number format candidates FP4 variant
- Split each matrix into blocks & assign the optimal format per-block

## Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

- static vs dynamic scaling


[Optimal Clipping and Magnitude-aware Differentiation (OCTAV)](https://arxiv.org/abs/2206.06501) is a method presented at ICML2022 designed to improve quantization-aware training (QAT) by efficiently determining the clipping scalar that minimizes the mean square error (MSE) between full-precision and quantized weights.

An alternative is using KL divergence to measure kMax and kMin

E4M3 for weight and activation tensors
E5M2 during gradient tensors

## References

[A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)

[Neural Network Quantization & Number Formats From First Principles](https://newsletter.semianalysis.com/p/neural-network-quantization-and-number)

[FP8 versus INT8 for efficient deep learning inference](https://arxiv.org/pdf/2303.17951)

[Microscaling Data Formats for Deep Learning](https://arxiv.org/pdf/2310.10537)

[Fixed Point Arithmetic in DSP](https://schaumont.dyn.wpi.edu/ece4703b20/lecture6.html)

[INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](https://arxiv.org/html/2510.25602v1)

[Floating point on NVIDIA GPU](https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf)

[Model Quantization in Deep Learning](https://www.ai-bites.net/model-quantization-in-deep-learning/)

[Quantization for Neural Network](https://leimao.github.io/article/Neural-Networks-Quantization/)

[Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning](https://www.usenix.org/system/files/atc21-zhou.pdf)

[Model Quantization: Concepts, Methods, and Why It Matters](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/)

[Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)

[RISC-V Composable Extensions for MX Microscaling Data Formats for AI Tensors: Part One: Introduction to MX Data](https://fpga.org/category/microscaling-mx-formats/)

[Floating-Point 8: An introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

[Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)

<!-- TODO How did DeepSeek do it -->

[DeepSeek Technical Analysis - FP8 Training](https://dataturbo.medium.com/deepseek-technical-analysis-5-fp8-training-ff34768727b8)