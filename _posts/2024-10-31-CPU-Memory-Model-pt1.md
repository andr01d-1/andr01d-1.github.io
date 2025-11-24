---
layout: post
title: "CPU Memory Model Part 1"
categories: CPU Memory Model
---

[A memory consistency model is a contract between the hardware and software. The software promises to only reorder operations in ways allowed by the model, and in return, the software acknowledges that all such reorderings are possible and that it needs to account for them](https://jamesbornholt.com/blog/memory-models/)


[Modern Multi-core CPUs often share their LLC (Last Level Cache, most often the L3) between its cores, and as a result is often placed well apart from each individual core](https://www.quora.com/Can-someone-show-me-a-picture-of-a-level-3-cache-and-how-far-is-it-from-the-CPU)

<!-- ### Store and Forward Buffer -->
### [*Store Buffer and Forwarding*](https://stackoverflow.com/questions/24176599/what-does-store-buffer-forwarding-mean-in-the-intel-developers-manual)

What is a [_Store Buffer_](https://stackoverflow.com/questions/11105827/what-is-a-store-buffer)?

How are they related to [_memory barriers_](https://dannas.name/2020/06/19/memory-barriers).


<p align="center">
    <img src="https://www.computerhope.com/issues/pictures/cpu-cache-die.png"/>
</p>

[Total store ordering (TSO)](https://pages.cs.wisc.edu/~markhill/papers/primer2020_2nd_edition.pdf) mostly preserves the same guarantees as sequential consistency, except that it allows the use of store buffers. These buffers hide write latency, making execution **[significantly faster](https://courses.grainger.illinois.edu/cs533/sp2025/reading_list/2b.pdf)**

Each core has its own _store buffer_ to

```
decouple execution from commit to L1d / cache misses,
and allow speculative exec of stores without making speculation visble in coherent cache.
```

ARM memory model is a form of _weak_ ordering, allows almost any operation to be reordered, which enables a variety of hardware optimizations but is also a nightmare to program at the lowest levels. 


Modern CPU's L1 cache usually is divided into two distinct parts

**L1 Instruction Cache(L1i)** 
and 
**L2 Data Cache (L1d)**

More design and implementation details in Futher Reading section

In addition to L1 cache, CPUs also have data-related components closer to the core, a `Load Buffer` and a `Store Buffer`. These buffers handle pending memory operations before they reach the cache or main memory. 
<span style="color:blue">(Blue connections in below diagram)</span>.

<div style="position:relative; width:100%; height:100vh;">
    <iframe src="https://mdsattacks.com/diagram.html" frameborder="0" style="position:absolute; top:0; left:0; width:100%; height:100%;"></iframe>
</div>

<div style="text-align:center; margin-top:10px;">
    <p>Interactive guide to speculative execution attacks hosted on <a href="https://mdsattacks.com/diagram.html" target="_blank">www.mdsattacks.com</a> from <a href="https://www.synkhronix.com/about/" target="_blank">micro-architectural attack researcher Stephan Van Schaik</a>.</p>
</div>

Two buffers are part of `out of order engine`.

### Store Buffer Size in Intel Archiectures

Intel Skylake architecture, typically features a `store buffer` with 56 entries. This size is optimized for balancing performance and hardware constraints. For example:
- **Skylake Architecture**: A 56-entry `store buffer` achieves 98.1% of the performance of [an ideal store buffer implementation without stalls](https://microarch.org/micro53/papers/738300a568.pdf).
- **Energy-Efficient Designs**: In scenarios with reduced store buffer sizes (e.g., 20 entries for energy efficiency), advanced techniques like [Store-Prefetch Burst (SPB)](https://microarch.org/micro53/papers/738300a568.pdf) can maintain comparable performance to larger buffers.

This demonstrates that while the `store buffer` size can vary, its design reflects trade-offs between performance, energy efficiency, and hardware complexity.

---

### Load Buffers vs. Store Buffers

- **Out-of-Order Execution**: `Load buffers` handle speculative reads, allowing CPUs to execute instructions out of order while hiding memory latency. This requires higher capacity to manage more in-flight loads. Both load and store buffers play critical roles in enabling out-of-order execution in modern CPUs. `Load buffers` allow speculative reads without waiting for prior instructions to complete, while `store buffers` allow writes to be deferred until they can be safely committed.

- **Memory Latency Mitigation**: Techniques like `Load Wait Buffers (LWB)` further optimize load handling by temporarily removing long-latency loads from the load/store queue, freeing up space for other instructions.

- **Memory Ordering**: `Store buffers` can introduce subtle issues with memory ordering because writes may appear out of order from the perspective of other cores. This is why CPUs implement mechanisms like memory barriers or fences to enforce proper ordering when required.

`Store buffers` hold write operations temporarily before committing them to the cache or main memory. Once data is committed from the store buffer to the cache, it becomes visible to other cores or processors in a multi-core system. This visibility is managed through cache coherence protocols e.g. [(MESI, MOESI)](https://developer.arm.com/documentation/den0013/latest/Multi-core-processors/Cache-coherency/MESI-and-MOESI-protocols), ensuring consistency across cores.

`Store buffers` are structured as queues where each entry corresponds to a write operation (address + value). The size of a store buffer is typically measured in terms of entries becuase each entry represents a discrete write transaction.

The use of entries rather than bits allows CPUs to manage multiple writes efficiently, regardless of the size of individual writes.

The reasoning behind smaller store buffers is that writes can be delayed or batched without significantly impacting performance, whereas reads are often latency-sensitive and need immediate attention.

The disparity in size reflects the architectural focus on optimizing read-heavy workloads.

---

<!-- Improving Instruction Per Cycle (IPC) is a central goal in microarchtecture optimization -->


When hyper-threading is enabled, resources such as the store buffer must be divided equally between two logical cores for use.  

The following resources are shared between two threads running on the same physical core:  
- Cache  
- Branch prediction resources  
- Instruction fetch and decoding units  
- Execution units  

Hyper-threading provides no performance advantage if any of these shared resources become a limiting factor for processing speed.

---

After a CPU operation modifies data, it doesn't immediately write to the L1d cache. Instead, it first writes to the **store buffer** and then flushes to the L1d cache as soon as possible. For example, in this code:

```c
int sum = 0;
for (int i = 0; i < 10086; ++i) {
  sum += i;
}
```

After each assignment to `sum`, the value isn't immediately written to L1d—it may reside in the store buffer for some time. Subsequent assignments might merge in the store buffer before flushing to L1d. This optimization improves efficiency by allowing the **speculator** to execute multiple operations (e.g., merging `+1` and `+2` into `+3`) before writing to the cache, reducing cache access time.

### Store Buffers and Reordering  
Store buffers can cause "out-of-order" effects. Consider this example:  

```c
// Global variables
int a = 0;
int b = 0;

// Thread 1                  |           // Thread 2
t1:                          |           t2:
a = 1;                       |           b = 1;
if (b == 0) {                |           if (a == 0) {
  // Do something            |             // Do something
} else {                     |           } else {
  goto t1; // Retry          |             goto t2; // Retry
}                            |           }
```

Both threads might simultaneously execute the "Do something" logic. 
- When Thread 1 writes `a = 1`, the value remains in its store buffer, making it invisible to Thread 2 (which still sees `a == 0`). 
- Similarly, Thread 2's `b = 1` stays in its store buffer, so Thread 1 sees `b == 0`. This violates mutual exclusion.

**Root Cause**: Without synchronization, store buffers break the `happens-before` relationship. Violating mutual exclusion assumptions in algorithms.

**Peterson’s/Dekker’s algorithms** rely on interleaving assumptions (e.g., `a = 1` being globally visible before checking `b`). `Store buffers` break these assumptions, making such algorithms unsafe on modern CPUs.

This demonstrates why **Peterson's and Dekker's algorithms** (classic mutual exclusion approaches) fail on modern CPUs. To fix this, replace `a` and `b` with **atomic variables**, which enforce memory model guarantees (e.g., sequential consistency). Atomic operations ensure writes are flushed from store buffers and propagated to other cores, making updates visible immediately. 

#### How It Works:

- Flushes the store buffer: Atomic writes (with `memory_order_seq_cst`) ensure immediate visibility by flushing the buffer.
- Enforces ordering: Prevents reordering of instructions around the atomic operation.
- Hardware support: Uses cache coherency protocols (e.g., MESI) to invalidate other cores’ cached values.

#### Additional Considerations
- Memory Ordering: Atomics allow weaker orderings (e.g., `memory_order_acquire`/`release`) for performance gains while still ensuring synchronization.
- Compiler Reordering: Modern compilers may reorder non-atomic operations; atomics also act as compiler barriers.
- Real-World Impact: This explains why low-level lock-free code often requires careful use of atomics or fences (e.g., in OS kernels or databases).

---

## Instruction Reordering

Before discussing memory order/barrier/fence, we need to understand the concept of **instruction reordering**, which occurs in two primary forms:  
1. **Compile-time reordering**: Performed by compilers to optimize register usage (e.g., reducing load/store operations).  
2. **Runtime reordering**: Done by CPUs to speculatively execute instructions and minimize pipeline stalls.  

Both aim to improve efficiency while adhering to a fundamental rule:  
> *["Thou shalt not modify the behavior of a single-threaded program."](https://preshing.com/20120625/memory-ordering-at-compile-time/)*  

This means reordering must preserve logical correctness in single-threaded execution. However, issues arise in **multi-threaded/multi-core** environments where memory visibility order changes due to reordering.

---

### Example: Multi-Threaded Visibility Issue  
```cpp
int msg = 0;
int ready = 0;

// Thread 1
void foo() {
  msg = 10086;
  ready = 1;  // Compiler/CPU may reorder these writes
}

// Thread 2
void bar() {
  if (ready == 1) {
    std::cout << msg;  // May output 0 unexpectedly
  }
}
```
Here, Thread 2 might see `ready == 1` before `msg` is updated, violating expectations. This necessitates **memory barriers** to enforce ordering.

---

#### Compile-Time Reordering  
Observe compiler optimizations in action:  
```cpp
// Original C++             | // Compiled Assembly (GCC -O2)
int a = 0;                  | 
int b = 0;                  | 
void foo() {                | foo():
  a = b + 1;                |   mov eax, DWORD PTR b[rip]
  b = 5;                    |   mov DWORD PTR b[rip], 5 
}                           |   add eax, 1 
                            |   mov DWORD PTR a[rip], eax 
                            |   ret 
```
The compiler reorders instructions to optimize register usage. To prevent this:  
```cpp
void foo() {
  a = b + 1;
  std::atomic_thread_fence(std::memory_order_release); // Enforce order
  b = 5;
}
```
Resulting assembly now follows source order. Similar optimizations occur for redundant assignments (e.g., eliminating intermediate writes).

Herb Sutter talks about considerations of compiler reordering, please see [Atomic Weapons 1 and 2](https://www.youtube.com/watch%3Fv%3DA8eCGOqgvH4) in Further Reading

---

### Key Takeaways  
- **Reordering**: Critical for performance but problematic in concurrency.  
- **Memory Barriers**: Tools like `std::atomic_thread_fence` enforce ordering constraints.  
- **Volatile**: Insufficient for multi-threaded synchronization (only prevents compiler reordering, not CPU reordering).  

This interplay between compiler and CPU optimizations underpins the need for explicit memory ordering primitives in concurrent programming.

---

### Runtime Reordering

Different CPU architectures (manufacturers) adopt varying strategies for runtime reordering. This involves another topic: the **hardware memory model** of CPUs. This note provides a detailed introduction to this topic.

1. **Hardware Memory Models Differ Across CPU Architectures**  
   The strategies/limitations for instruction reordering vary depending on the hardware memory model of the CPU architecture.

2. **Two Main Types of Hardware Memory Models**  
   - **Weak Hardware Memory Model**: Found in ARM, Itanium, PowerPC architectures. These models impose fewer restrictions on instruction reordering. The complex architecture and implementation of CPU caches contribute significantly to the complexity of memory visibility (order). However, the reduced restrictions allow software developers more flexibility to optimize performance.  
   - **Strong Hardware Memory Model**: Found in x86-64 family CPUs (Intel and AMD). These models impose many restrictions on instruction reordering.  
     - A strong hardware memory model ensures that every machine instruction implicitly has acquire and release semantics. Consequently, when one CPU core performs a sequence of writes, all other CPU cores observe those values changing in the same order they were written.  
     - For example, if a core executes an instruction stream with `n` memory write operations, when it reaches the `k-th` write instruction, all previous `k-1` writes can be observed by other cores in the same order as they occur on the original core. Ensuring this consistency incurs performance overhead due to the presence of CPU caches, which make visibility more challenging. The strong model also limits optimization to some extent.

3. **Even Within Weak or Strong Models, Differences Exist**  
   Since instruction sets vary widely, even within the same type (weak or strong), strategies differ significantly across CPU architectures.

4. **Rules for Instruction Reordering**  
   Regardless of the CPU architecture, instruction reordering follows specific rules documented in each CPU's manual. These manuals are essential reading for system (OS) developers.

5. **Intel's x86 Series Philosophy**  
   Intel's approach is that as long as individual instructions execute fast enough, optimization concerns can be minimized.

The concept of "acquire and release semantics" mentioned above will be elaborated in subsequent sections.

---

#### Gotcha! Runtime Reordering

Next, we need to prove that runtime reordering does exist. It's important to understand that runtime reordering is difficult to capture because it cannot be analyzed statically—it requires actual execution and may not occur consistently during runtime since the CPU's instruction stream varies at different moments.

##### Code Example: Detecting Runtime Reordering

The following code snippet demonstrates how threads can be used to detect runtime reordering:

```cpp
void test() {
  for (;;) {
    int msg = 0;
    int ready = 0;
    // Thread 1
    std::thread t1([&msg, &ready] {
        msg = 10086;
        ready = 1;
      }
    );

    // Thread 2
    std::thread t2([&msg, &ready] {
        if (ready == 1 && msg == 0) {
          std::cout  random(1, 100000000);
  for (;;) {
    sem_wait(&begin_sema1); // Wait for signal
    
    while (random(rng) % 8 != 0) {} // Random delay increases chances of capturing reordering

    X = 1;
    asm volatile("" ::: "memory"); // Prevent compiler reordering explicitly
    r1 = Y;

    sem_post(&end_sema); // Notify transaction complete
  }
}
```

##### Main Function:
```cpp
int main(void) {
  sem_init(&begin_sema1, 0, 0);
  sem_init(&begin_sema2, 0, 0);
  sem_init(&end_sema, 0, 0);

  std::thread thread1([] { thread1_func(nullptr); });
  std::thread thread2([] { thread2_func(nullptr); });

  int detected = 0;
  for (int iterations = 1; ; ++iterations) {
    X = Y = detected = r1 = r2 = 0;

    sem_post(&begin_sema1);
    sem_post(&begin_sema2);

    sem_wait(&end_sema);
    sem_wait(&end_sema);

    if (r1 == 0 && r2 == 0) {
      std::cout  ai = {0};
    }
  }
  ai.compare_exchange_weak(a, value);
}
```

Key Considerations

  - Memory Barriers: Missing in this code, allowing reordering
  - Synchronization: Semaphores coordinate thread starts but don't prevent memory reordering
  - Architecture Dependence: Detection rate would vary between x86 (stronger memory model) vs ARM (weaker model)

This pattern is commonly used to demonstrate:

  - Why atomic operations need explicit memory ordering constraints
  - How compilers/CPUs can reorder instructions
  - The importance of memory models in concurrent programming


Intel’s manual explains how locked atomic operations guarantee synchronization across processors



## Further Reading

[Size of store buffers on Inteal hardware? What exactly is a store buffer?](https://stackoverflow.com/questions/54876208/size-of-store-buffers-on-intel-hardware-what-exactly-is-a-store-buffer)

[How to interpret LLC load misses from perf stats](https://stackoverflow.com/questions/52138985/how-to-interpret-llc-load-misses-from-perf-stats)

<!-- [How can I profile a kernel over time with CUPTI](https://stackoverflow.com/questions/70403600/how-can-i-profile-a-kernel-over-time-with-cupti) -->

[CPU Cache Explained](https://hothardware.com/news/cpu-cache-explained)

[Developing intuition when working with performance counters](https://easyperf.net/blog/2019/07/26/Developing-intuition-when-working-with-performance-counters)

[McGill University CS273 No.18 Data and Instruction Caches](https://www.cim.mcgill.ca/~langer/273/18-notes.pdf)

[What is meant by data cache and instruction cache](https://stackoverflow.com/questions/22394750/what-is-meant-by-data-cache-and-instruction-cache)

[How and where are instructions cached](https://stackoverflow.com/questions/59762711/how-and-where-are-instructions-cached)

[C++ and Beyond 2012: Herb Sutter Atomic Weapons 1 of 2](https://www.youtube.com/watch?v=A8eCGOqgvH4)

[C++ and Beyond 2012: Herb Sutter Atomic Weapons 2 of 2](https://preshing.com/20120625/memory-ordering-at-compile-time/)

[Gavin Chou C++ memory ordering](https://gavinchou.github.io/summary/c++/memory-ordering)

[Gallery of processor cache effects](https://igoro.com/archive/gallery-of-processor-cache-effects/)

[Occurrence of instructions among C/C++ binaries in Ubuntu 16.04](https://x86instructionpop.com/)

[Paper: Memory Barriers, A hardware view for Software hackers](http://codelabs.ru/reading/whymb.2010.07.23a.pdf)

[How does the x86 TSO memory consistency model work when some of the stores being observed from store-forwarding](https://stackoverflow.com/questions/69925465/how-does-the-x86-tso-memory-consistency-model-work-when-some-of-the-stores-being)