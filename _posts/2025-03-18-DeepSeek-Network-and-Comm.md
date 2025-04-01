---
layout: post
title: "DeepSeek Code Reading and Study 02"
categories: GPU, LLM, Distributed, Network, RDMA
---

This analysis combines DeepSeek's technical report and 3FS source code to examine its network communication module, particularly focusing on RDMA implementation.

## System Overview  
The 3FS cluster consists of 180 storage nodes, each equipped with:
- [`2×200 Gbps` InfiniBand NICs](https://network.nvidia.com/files/doc-2020/wp-introducing-200g-hdr-infiniband-solutions.pdf)  
- 16×14TiB NVMe SSDs  
Achieving **6.6 TiB/s aggregate read throughput** across all nodes.  

Four core components form the architecture:
1. **Cluster Manager**  
2. **Metadata Service**  
3. **Storage Service**  
4. **Client**  

All components communicate via RDMA networks.

## Network Module Design Highlights  
### Performance Optimization  
- **[Full-stack folly coroutines](https://developers.facebook.com/blog/post/2021/10/14/async-stack-traces-c-plus-plus-coroutines-folly-walking-async-stack/)** for asynchronizing I/O-bound operations  
- **C++20 features** (e.g., `std::span`) for [zero-copy data handling](https://brevzin.github.io/c++/2018/12/03/span-best-span/)  
- **Batch processing** with [`RDMAPostCtx`](https://github.com/deepseek-ai/3FS/blob/main/src/common/net/ib/IBSocket.h#L245) minimizing [CQE (Completion Queue Entries) generation](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/key+concepts)
- [`sq_sig_all=0`](https://www.ibm.com/docs/en/aix/7.1?topic=management-ibv-create-qp-ibv-destroy-qp) configuration reducing completion queue events  

### Architectural Flexibility  
- Decoupled client/server implementation  
- No dependency on specific RPC frameworks  
- Distributed buffer management without centralized metadata  

### Engineering Refinements  
- Multi-NIC parallelization  
- Custom SERDE encoding/decoding service  
- Transport sharding in [`TransportPool`](https://github.com/deepseek-ai/3FS/blob/main/src/common/net/TransportPool.h)

## Core Communication Classes  
### IBSocket Implementation  
Located in [`src/common/net`](https://github.com/deepseek-ai/3FS/tree/main/src/common/net), key methods include:  
```cpp
rdmaRead() → CoTryTask
rdmaWrite() → CoTryTask
```
- Batches RDMA requests via `RDMAReqBatch`  
- Uses `IBV_WR_RDMA_READ/WRITE` for unilateral operations  
- Implements graceful closure with `IBV_WR_RDMA_WRITE_WITH_IMM`

### RDMA Device Management  
`IBDevice` class handles:  
1. Device enumeration  
2. Protection domain allocation  
3. Queue Pair (QP) creation  
```cpp
qpCreate() → verbs-based QP initialization
qpReadyToRecv() → state transition to READY
```

### Event-Driven Architecture  
1. **Listener**  
   - Manages multiple NICs via `EventBase` threads  
   - Uses [`folly::blockingWait`](https://github.com/facebook/folly/blob/main/folly/experimental/coro/BlockingWait.h) for socket creation  
2. **IOWorker**  
   - Processes I/O tasks through - Transport sharding in [`TransportPool`](https://github.com/deepseek-ai/3FS/blob/main/src/common/net/TransportPool.h)
  
   - Implements address-based sharding  
3. **EventLoop**  
   - Epoll-based event notification  
   - Handles 2,000+ concurrent connections  

## RDMA I/O Workflow  
### Write Path  
1. **ReliableUpdate Module**  
   - Chain replication via CRAQ protocol  
   - Maintains committed/pending chunk versions  
2. **StorageOperator::write**  
   ```cpp
   ReliableUpdate::update() → RDMA_WRITE
   TargetA → TargetB via ReliableForwarding
   ```
3. **Batch Optimization**  
   - Last WR in batch uses [`IBV_SEND_SIGNALED`](https://www.ibm.com/docs/en/aix/7.1?topic=management-ibv-post-send)
   - Vectorized `RDMAPostCtx` processing  

### Read Path  
1. **[StorageOperator::batchRead](https://github.com/deepseek-ai/3FS/blob/main/src/storage/service/StorageOperator.h#L70)**  
   - Server-initiated RDMA_WRITE to client  
2. **[AioReadWorker](https://github.com/deepseek-ai/3FS/blob/main/src/storage/aio/AioReadWorker.h)**  
   - Manages NVMe SSD access  
   - Integrates Serde encoding for RDMA buffers  

## Memory Management  
### [RDMABufPool](https://github.com/deepseek-ai/3FS/blob/main/src/common/net/ib/RDMABuf.cc)
- Decentralized buffer allocation  
- Asynchronous allocation via `CoTask`  
- Buffer lifecycle tied to coroutine scope  

### [Mooncake](https://github.com/kvcache-ai/Mooncake/tree/main) Comparison  
| Feature               | 3FS                    | Mooncake TransferEngine  |
|-----------------------|------------------------|--------------------------|
| Buffer Registration   | On-demand              | Pre-registered           |
| Metadata Dependency   | None                   | Requires MetaService     |
| Use Case              | General storage        | LLM inference-specific   |

## Folly Coroutine Integration  
- **Full-stack async** from I/O to memory allocation  
- `folly::coro::Baton` synchronizes CQE polling  
- Coroutine chaining pattern:  
  ```cpp
  CoTryTask<> rdmaBatch() {
    auto posts = splitRequests();
    co_await collectAllRange(postTasks);
  }
  ```

## Optimization Opportunities  
1. **Error Handling**  
   - Current implementation lacks WR retry logic  
2. **Lock Contention**  
   - `EventHandler` mutex in critical path  
3. **Storage Integration**  
   - Potential SPDK adoption for user-mode NVMe  

3FS's industrial-grade RDMA, large-scale model training, coroutine-driven archiecture and decentralized resource management offer a paradigm for high-throughput storage systems