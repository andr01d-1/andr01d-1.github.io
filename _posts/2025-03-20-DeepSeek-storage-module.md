---
layout: post
title: "DeepSeek Storage Module"
categories: GPU, LLM, Distributed, Storage
---

## Technical Analysis of 3FS Storage Module

## I. Architectural Overview
**Core Components**:
- **StorageService**: Central management layer with three fundamental capabilities:
  1. **Unified Storage Pooling**: Implements chunk-based resource virtualization
  2. **RDMA-Optimized Data Path**: Provides high-performance network I/O
  3. **Chained Replication Support**: Enables distributed fault tolerance

Each physical disk manages multiple virtual StorageTargets through DiskUnit abstraction. ChunkEngine serves as the core execution unit for storage management.

## II. Space Management Mechanism

### Structure

 - The Allocator consists of 11 instances responsible for allocating chunks ranging from 64KB to 64MB
 - Each Allocator manages 256 Files for actual data storage. Files are logically organized into Groups, with each Group containing 256 Chunks managed using bitmap indexing

### Allocation Process

#### Group Allocation

Groups are allocated in a specific sequence (as shown in the diagram) to ensure blanaced file sizes. The Engine requets new storage blocks through Allocator::allocate. The process involves:

- Checking existing storage blocks for availability
- Invoking `ChunkAllocator` and `GroupAllocator` for new blocks/Groups when needed
- Requesting new Group allocation through GroupAllocator when
  - No `active_group` exists
  - All existing Groups are `full_group`

#### Chunk Allocation Workflow

- Allocator Selection: Choose the appropriate allocator baesd on requested size
- Group Selection:
  - Prioritize least-idle existing Group
  - Allocate new Group if none available
  - Select Chunk from chosen Group
- System Update:
  - Update Engine's `chunk_id-to-chunk` mapping
  - Persist both mapping relationships and allocator allocation information

The technical description explains a memory management system focusing on chunk allocation strategies and reference counting mechanisms. Here's the key analysis:

## Fragmentation Management
### Group Allocation Optimization
Steps a and b optimize internal fragmentation through:
1. **Slab-like allocation** (similar to [OS slab allocators](https://en.wikipedia.org/wiki/Slab_allocation))
2. **Greedy hole-filling** in groups to maximize space utilization
This strategy requires:
- Fixed chunk size patterns from upper layers for optimal matching
- Tradeoffs between:
  - **Stability**: Predictable allocation sequence (shown in diagram)
  - **Flexibility**: On-demand allocation could cause performance spikes

The approach shows best performance in scenarios with:
- Minimal delete/overwrite operations
- Sequential group allocation patterns
While NVMe SSD's parallel access mitigates locality issues from scattered holes[15]

## Index Persistence (Step c)
Maintains stable upper-layer indexing through:
- **Autonomous chunk mapping** via local RocksDB storage

## Autonomous Resource Management
Each Target implements self-contained resources with:
- Independent MetaDB instances (RocksDB per target)
- Localized metadata management
- Decentralized allocation/release operations

## Reference Counting Mechanism
Implements RCU-style memory management:

Key characteristics:
1. **Atomic counters** for thread safety
2. **Position-based tracking** instead of object-level
3. **Automatic group recycling** when refcount reaches zero

This design balances:
- **Fragmentation control** through grouped allocation
- **Flexibility** via reference-counted autonomous management
- **Performance** leveraging NVMe SSD parallelism

The architecture shows similarities with modern memory management systems combining slab allocation and reference counting, while addressing storage-specific requirements through localized metadata management and SSD-optimized access patterns.

*Note: Original implementation uses Rust's async/await for I/O workers and achieves ~8μs P99 latency for 4KB reads in lab tests.*
