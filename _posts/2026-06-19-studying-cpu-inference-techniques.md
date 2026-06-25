---
layout: post
title: "Studying CPU Inference Techniques"
categories: CPU Inference NN
---

I was curious about [what happened](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.reddit.com/r/chess/comments/rhs3zn/hey_what_ever_happened_to_alpha_zero/) to [Alpha Zero](https://en.wikipedia.org/wiki/AlphaZero)


As a research and experimental project, Alpha Zero is no longer being developed. Modern Chess engines like [StockFish](https://stockfishchess.org/) are considered stronger than the revolutionary Alpha Zero.

One architecture is frequently mentioned in many online Chess engine discussion is [NNUE (Efficiently Updatable Neural Networks)](https://www.chessprogramming.org/NNUE). It is the core ["evaluation function"](https://en.wikipedia.org/wiki/Evaluation_function).

It sits right in the intersection of search algorithm design, neural network design, reinforcement learning and Computer Architecture edge inference performance optimization.


Reference 1[^1] covers a few interesting topics around the engine, including grandmasters' perspectives on Neural-Network based chess engines.

In this note, we will focus on design and optimization that is more generalizable towards low latency inference network applied to mission critical applications at the edge like autonomous vehicles, robotics and more. 


## Search + Neural Network Evaluation

Two major design approaches for game trees are 

- Alpha-Beta Search/Pruning (AB) + NNUE
- Monte Carlo Tree Search (MCTS) + DNN

Alpha-Beta pruning (AB) and Monte Carlo Tree Search (MCTS) are both algorithms used to navigate game trees, but they approach the problem from opposite directions.

AB is an exhaustive, deterministic search that evaluates every possible move sequentially, while MCTS is a probabilistic, heuristic search that uses random sampling to explore deeper, more promising lines of play.

A practical consideration: [MCTS is easier to parallelize](https://ai.stackexchange.com/questions/37419/the-reason-behind-using-mcts-over-alpha-beta-pruning-in-alphazero)



## Design Decision
 
NNUE is a [4 layer (1 input + 3 dense)](https://www.chessprogramming.org/Stockfish_NNUE#NNUE_Structure) integer only network. Just over 82K parameters. 


From its name the *efficiently updatable*, instead of doing a full matrix multiplication over the entire first layer, the engine simply subtracts the weight contribution of the pieces that moved, and adds the contributions of the pieces that landed. Instead of a full `matmul` which often requires dedicated NN accelerators.

A beautifully illustrated network input/output configuration can be found in a Rust chess engine implementation note[^4]

<p align="center">
    <img src="https://slama.dev/prokopakop/nnues-and-where-to-find-them/network.svg" />
</p>


## What about GO?

A Go game engine maintainer explains the difference of fundamental natures of Go and Chess. A shallow network like NNUE would be difficult to learn more abstract and long term features. [^5]


## Optimization details

Number of Non-Zero elements (NNZ) related operations are often heavily optimized in hardware[^10]

It looks like the creator of Rust based chess engine ([viridithas](https://github.com/cosmobobak/viridithas)) uses a compressed format[^9] in sparse matrix multiplication (SpMM) SIMD implementation[^6]. 


## Take away

It is an extreme example of problem algorithm, hardware and software co-design[^3]. The creators intentionally leverage 
- Chess specific features to redesign the algorithm for increasing computation arithmetic intensity
- Reducing memory operations (load, store)
- quantisation[^7]
- Carefully redesign ReLU for SIMD ops without losing accuracy
- Avoiding register spilling
- keeping all vectorized operations cache-aligned

## Further thoughts

By keeping everything on the CPU side, it is reported that NNUE engine can achieve sub $\mu $second per thread evaluation.

For Modern GPUs going through PCIs, the packing and round trip would take 1 to 5 $\mu$ seconds. It wouldn't suite for extreme latency sensitive application.

For embedded accelerators like Rockchip's NPU in low cost SoC RV1109, going bare metal with it NPU is unsupported, if we use its default Linux setup, I would imagine high overhead when going through OS. 

(TODO: do an actual bench of NPU/RKNN round trip time test)

What are SoC designer doing to further reduce this overhead? Something to learn about for our next writeup.



## References

[^1]: [Development of Neural Network Chess Engines](https://beuke.org/nnue/)
[^2]: [Chess Wiki NNUE](https://www.chessprogramming.org/index.php?title=NNUE)
[^3]: [NNUE Performance Improvements](https://asteri.sm/files/2024-06-01-nnue)
[^4]: [NNUEs, and Where to Find Them](https://slama.dev/prokopakop/nnues-and-where-to-find-them/)
[^5]: [Is it worth using NNUE to replace FPU?](https://github.com/lightvector/KataGo/issues/379)
[^6]: [Deep NNUE](https://asteri.sm/files/2024-08-17-multilayer)
[^7]: [Micronote on microscaling](https://andr01d-1.github.io/vlsi/microscaling/micro-architecture/2026/01/09/micronote-on-microscaling.html)
[^8]: [Counting leading zeros from software to hardware](https://andr01d-1.github.io/low-level-gpu/2024/09/17/counting-leading-zeros.html)
[^9]: [Grouped linear and LLM](https://andr01d-1.github.io/sparse-tensor-core/llm/grouped-linear/2026/05/15/grouped-linear-and-llm.html)
[^10]: [Efficient Sparse Matrix-Vector Multiplication on x86-Based Many-Core Processors](https://faculty.cc.gatech.edu/~echow/pubs/ics26-liuPS.pdf)