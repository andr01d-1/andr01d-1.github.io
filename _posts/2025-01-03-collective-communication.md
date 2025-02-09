---
layout: post
title: "Collective Communication"
categories: communication
---


Large machine learning model, especially deep neural networks, often require distributed training across multiple GPUs or nodes due to their size and computational demands.
The distribution necessitates efficent communication between processing units.

*Collective Communication* operations are crucial for distributed machine learning.

## Terms

### Synchronization

- Barrier: All processes wait until everyone reaches a specific point in the program

### Data Movement

- Broadcast: A source process sends identical data to all other processes
- Scatter: A source process sends a distinct message to all other processes
- Gather: This operation is the reverse of scatter
- All-to-all broadcast: Every process communicates the same data to every other process
- All-gather: Collects data from all processes and distributes it to all processes
- All-to-all personalized exchange: Every process communicates the same data to every other process

### Collective Computation

- Reduce: Combines data from all processes using a specific operation (e.g., sum, max, min).
- All-reduce: Similar to reduce, but the result is distributed to all processes
- Scan (or Prefix-sum): Performs a cumulative operation across ordered processes

#### [Visualization](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node64.html)

![Visualization](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/coll-fig1.gif)

## MPI 

or Message Passing Interfaces, is a standarized and portable communication protocol designed for parallel computing  

It is also a programming model 

### MPI (Libraries) Architecture

A figure illustrates the architecture for the popular MPI libraries from [Azure Virtual Machine Setup MPI guide](https://learn.microsoft.com/en-us/azure/virtual-machines/setup-mpi)

![HPC Applications](https://learn.microsoft.com/en-us/azure/virtual-machines/media/hpc/mpi-architecture.png)


### Terms

`Processs`: actual instances of the program running
`Groups`: logical `groups` of `processes`
`Rank`: A process is identified by its `rank` (An integer in the range `[0, N - 1)` where Ni s the size of group)
`Communicators`: objects handle communication between processes.
`Intra-communicator`: handles processes within a single group
`Inter-communicator`: handles communication between two distinct groups

[`The rank of a process is always relative to a group`](https://stackoverflow.com/questions/5399110/what-is-the-difference-between-ranks-and-processes-in-mpi)





## Networking

[Remote Direct Memory Access (RDMA)](https://en.wikipedia.org/wiki/Remote_direct_memory_access) 

Several main network protocol RDMA: 
InfiniBand protocol is a specific network architecture that implements RDMA

RMDA can be implemented on various network fabrics, including InfiniBand,
RDMA over Converged (RoCE) RDMA over an Ethernet Network
and 
iWARP (internet Wide Area RDMA Protocol) a protocol that allows using RDMA over TCP/IP 

InfiniBand typically offers the highest performance for RDMA operations due to its specialized hardware

[RoCE might approach similar performance levels but may have slightly higher latency](https://cloudswit.ch/blogs/roce-or-infiniband-technical-comparison/)

The same source argues,
```
At the physical layer, both RoCE and IB support 800G. However, PAM4's four distinct voltage levels is the superior option compared to Non-Return-to-Zero (NRZ) binary modulation, 
and Ethernet is the more cost-effective choice. RoCE is the clear winner
```

```
NVidia further develops, "SHARP" offloads collective communication operations--like all-reduce, reduce, and broadcast--from the server's compute engines to the network switches. 
By performing reductions (summing, averaging, and so on) directly within the network fabric.
```

From 

[Advancing Performance with NVIDIA SHARP In-Network Computing](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)

InfiniBand adapters typically use PCIe as interface to connect to host systems.

### NVLink

and Infiniband are complementary interconnect technologies often used together in HPC environments and data centers.

[NVLink](https://en.wikipedia.org/wiki/NVLink) is specifically designed for GPU-to-GPU and GPU-to-CPU connections within a server or across a limited number of servers.


## NCCL (Nvidia Collective Communications Library)

We will have a different writeup for NCCL implementation details and code work-through


<!-- ### Example

Seems more of graph level optimization?

### TODO 

Others over NCCL 

e.g. DeepSeek and Facebook? -->


### TODO: Common Network Topology

Spine-leaf or etc. 




## References and Further Reading

- Durato, J. and Yalamanchili, S. and Lionel, N. (2003). *Interconnection Networks: An Engineering Approach*. Morgan Kaufmann
- Dally, W. J. and Towles, B. (2004). *Principles and Practices of Interconnection Networks*. Morgan Kaufmann
- Rosen, R. (2013). Linux Kernel Networking: Implementation and Theory. Apress
- [RDMA Code Tutorial](https://github.com/jcxue/RDMA-Tutorial)
- [NCCL: High-Speed Inter-GPU Communication for Large-Scale Training](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/) (2021)
- [RoCE Networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/) (2024)
<!-- https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/ -->
<!-- https://news.ycombinator.com/item?id=41374663 -->

