---
layout: post
title: "Code as Data, Data as Code"
categories: misc
---

Modern CPUs employ a wide range of techniques, such as pipelining, branch prediction, and Out of Order execution (OOE), to optimize the control path and data flow. On the other hand, GPUs feature massively parallel machinery with compute units that are comparatively simpler. These unites are not [microcoded (in the traditional sense), nor do they utilize OOE, branch prediction, or superscalar capabilities, favoring instead the allocation of more silicon space for computation and high-speed registers.](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf).


<p align="center">
    <img src="https://www.legitreviews.com/images/reviews/1100/Fermi_Die.jpg" />
</p>

<p style="text-align: center;"><a href="https://www.legitreviews.com/nvidia-announces-cuda-gpu-architecture-fermi_1100">Fermi Die Shot</a></p>

Static optimization of machine learning models can be achieved either through manually-tuned low-level implementation or through multi-stage machine learning compiler automation, such as [XLA (Accelerated Linear Algebra)](https://openxla.org/xla/architecture#how_it_works). These compilers further refine the representations into actual machine instructions that are often specialized for specific hardware primitives, such as NVIDIA's [xMMA](https://gist.github.com/malfet/8ed6e5906a6ec7b9c6d779b27aa49a0e), Intel's DL Boost, and the instruction sets used in [Apple Silicon](https://eclecticlight.co/2024/03/01/apple-silicon-4-a-little-help-from-friends-and-co-processors/).

High level optimization like OPs fusion is essentially a result of [static data flow analysis](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html?highlight=data%20flow%20analysis#finalizing-the-operation-graph). _Data Flow_ is loosely expressed/encoded in the architecture and _connection weights_. While _Control flow_ is the runtime and dynamic decision making affected by _weights_ encoding.

Under this mental model, a very loose analogy we could draw between neural network OPs and classical processor operators.

Hashing historically execution info is a commonly used technique for implementing [branch prediction](https://www.quora.com/How-do-neural-branch-predictors-work-and-how-are-they-implemented). e.g. [GShare](https://github.com/IanBoyanZhang/HDLBits/blob/master/CS450/gshare.v), or [*Software Assisted Branch Prediction*](https://twitter.com/jonmasters/status/1399234979679330305) (or [neural networks](https://www.theregister.com/2016/08/22/samsung_m1_core/) to assist dynamic behavior for [some time](https://news.ycombinator.com/item?id=12340661)). We could consider this as a hardware/software coordinated heuristic for runtime optimization.

Classical Neural Networks do not inherently encode control flow in the same way that software programming (whatever that means) does, certain neural architectures and techniques allow for a form of control flow modeling. _Recurrent Neural Networks (RNNs)_ and their variants (like _LSTM_ and _GRU_) can model temporal sequences where the output at a given step depends on previous computations, which is a form of control flow.

The power of transformers comes from their self-attention layers, which can adapt their connection weights based on the context of the input. This is different from traditional networks with fixed connection weights after training. The self-attention mechanism allows transformers to be more efficient in processing information, which is why they can perform tasks that would require a much larger network if only fixed weights were used. While transformers do not use dynamic weights in the sense of weights that change after training, they do use a form of dynamic computation through the attention mechanism, which calculates weights based on the input data during both training and inference.

By dynamically adjusting the network _focus_, thereby guiding the processing and decision-making, _Attention_ mechanism could be seen as a flexible, data-driven form of control flow.

Perhaps the aforementioned thinking can be applied to hardware and software heurstics implementation for further optimizing networks' runtime behavior. 

Traditionally, the softmax function computes the exponentials of all input values and then divides each by the sum of all exponentials. This can be computationally expensive and memory-intensive, especially with large datasets or in real-time processing scenarios.
Instead of recalculating the sum of exponentials from scratch for each input, [The "online normalizer calculation" method](https://github.com/NVIDIA/online-softmax/tree/master) maintains a running sum and adjusts it incrementally as new data points are processed. This approach reduces redundant memory accesses, which are a common bottleneck in traditional softmax implementations

_IO awareness_ and _memory-efficiency_ are the core design principles of _Flash Attention_ and later _Paged Attention_.

What if we push one step further, instead of manually designing algorithms to relief memory pressure on hot paths, we have an online optimizer (a Neural Network?) to dynamically adjust _computation to IO ratio_ by changing accessing pattern while preserving statistically features? or by encoding "control flow" efficiently when [branch specialization](https://distill.pub/2020/circuits/branch-specialization/) happens?

PyTorch uses what they call [_Caching Memory Allocator_](https://pytorch.org/docs/stable/notes/cuda.html#memory-management), Roughly speaking, [it allocates a large chunk of GPU memory and implements a heap with it](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html). If needed they expose some knobs and functions to allow you to control it and observe the memory usage.

I wonder what Transformer Accelerator would be like, how it might leverage software to introduce more _"dynamic"_ behaviors. If this will enable a new type of networks that have _"dynamic architectures"_.

Turns out this isn't a new idea.

> As I have frequently emphasized since 1990, the connection strengths or weights of an artificial neural network (NN) should be viewed as its program. Inspired by Gödel's universal self-referential formal systems, I built NNs whose outputs are changes of programs or weight matrices of other NNs, and even self-referential recurrent NNs (RNNs) that can run and inspect and modify their own weight change algorithms or learning algorithms

- From [Juergen's Neural nets learn to program neural nets with fast weights—the first Transformer variants](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html)