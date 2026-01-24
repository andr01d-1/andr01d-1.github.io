---
layout: post
title: "GPU vs FPGA programming models"
categories: Heterogeneous Computing, FPGA, GPU, Programming Models
---

## [What Is An FPGA](https://www.youtube.com/watch?v=gUsHwi4M4xE)

This [visualization](https://www.youtube.com/watch?v=iHg0mmIg0UU) is probably the best visual guide I have encountered explaining how FPGA is different from our rudimentary mental model of single process processor being programmed.


Conceptually, an FPGA can be visualized as a giant virtual [breadboard](https://m.media-amazon.com/images/I/61mral12olL.jpg) or a field of [digital LEGOs](https://projectf.io/about/). It is covered with a "sea of configurable blocks" and a programmable network of switches can be wired together. The exact naming being used for these blocks are ["vendor terms"](https://www.reddit.com/r/FPGA/comments/fz981e/fpga_terminology_clarification/), we use [Xilinx](https://gab.wallawalla.edu/~larry.aamodt/engr433/xilinx_10/xilinx_10_gls.pdf)'s convention.

  - [Configurable Logic Blocks (CLBs)](https://digilent.com/blog/fpga-configurable-logic-block/): These are the basic building units of an FPGA
    - Lookup Tables (LUTs): Memory cells that store a pre-calculated list of outputs for any given set of inputs. 
    - Flip-Flops: The smallest storage elements, used to save logic states between clock cycles.
  - Programmable Interconnects: Also known as the routing fabric, this is a network of programmable wires and switches that connects the CLBs and other blocks, allowing for flexible circuit design

  - Input/Output Blocks (IOBs): These blocks serve as the interface, connecting the internal logic to external devices

  - Specialized Blocks: Modern FPGAs also include fixed-function blocks, such as Digital Signal Processors (DSPs), memory blocks, and embedded processor cores, for improved performance


<!-- ### Switch networks and Routing Fabric -->







## Abstraction Level



FPGAs and GPUs operate at fundamentally different levels of abstraction.

- GPUs: Rely on a fixed, high-level architecture optimized for parallel computing tasks
- FPGAs: Offer a much lower-level form of programmability, allowing you to create the digital hardware emulation circuit itself. This gives designers the flexibility to build a custom circuit (emulation) optimized for specific 

There is no ["runtime"](https://en.wikipedia.org/wiki/Execution_(computing)) on FPGA in the traditional sense. *We might spin up a [soft processor](https://en.wikipedia.org/wiki/Soft_microprocessor) or use the processor cores part of modern SoC FPGA. But there aren't really use FPGA in that sense anymore.*



<!-- The network 

There are a couple of  -->





<!-- When you 




How are CLBs connected to each other? 

2D mesh architecture by a programmable routing network. 


- SRAM-Based Programming Technology (it takes time to load the onboard distributed SRAM)
  - Flash based
- Mesh network



FPGA's 


Crossbar in GPU, NVLINK

Hardware  -->






## Programmability

A common mantra among FPGA engineers is that [they don't "code" ](https://news.ycombinator.com/item?id=43734311) - they describe hardware. A [Hardware Description Language (HDL)](https://en.wikipedia.org/wiki/Hardware_description_language), as its name suggests, allows defining physical circuits and connections. This is a fundamental departure from typical software development, where instructions are executed largely sequentially.

HDL is closer to a dataflow model language with its inherent concurrent nature, though it includes features to express sequential, control-flow-like behaviors for state machines and clocked logic. 

Many programming languages have explicit constructs to model [control-flow](https://en.wikipedia.org/wiki/Control_flow), but as a simulation instrument, different from software programming language ["HDL explicitly include the notion of time"](https://cs.stackexchange.com/questions/77532/what-does-it-mean-that-hdls-explicitly-include-the-notion-of-time).


It sorta resembles sheet music.

![sheet music](https://upload.wikimedia.org/wikipedia/commons/4/4c/CuiVil3_2p204.png)

Each *bar line* visually represents the passage of *time*.

- HDLs explicitly model timing through constructs that specify delays, event ordering and synchronization, enabling simulation.

- Music scores use timing signatures and note durations to organize (sound) events over measurable intervals

Multiple instruments can concurrently play, as long as they meet timing constraints.


The dominated languages like VHDL and (System)Verilog can model at [gate-level](https://www.chipverify.com/verilog/verilog-gate-level-modeling), [register-transfer-level](https://en.wikipedia.org/wiki/Register-transfer_level) and [behavior level](https://www.digikey.com/en/maker/tutorials/2025/behavioral-modeling-in-verilog-part-19). 


- Behavior level modeling is closer to the "software programming", we describe what a circuit does, or its algorithm, rather than how it is physically implemented. Behavioral models are used for fast simulation and verification of a system's overall function and are often not synthesizable into actual hardware

- Register-Transfer Level (RTL): This is the most common level for digital design and is the primary input for synthesis tools. An RTL description specifies the flow of data between registers and the logical operations performed on that data (Actual digital designers would frown upon this definition). The synthesis tool then maps this description to the actual gates and flip-flops of the target technology

- Gate level: This is the lowest and most detailed level of abstraction, where the design is described in terms of basic logic gates (such as AND, OR, and NOT) and their interconnections. This level is very close to the actual hardware implementation. Designers typically do not write gate-level code directly; instead, it is the output generated by synthesis tools from an RTL description. Gate-level models are used for post-synthesis verification and [static timing analysis](https://en.wikipedia.org/wiki/Static_timing_analysis)



```verilog
// Behavioral 4-to-1 multiplexer (mux)
module mux_4_to_1_beh (
    input  wire [3:0] d,         // 4 data inputs
    input  wire [1:0] sel,       // 2 select lines
    output reg        y          // 1-bit output
);

  // The always block is triggered by changes in any of its inputs.
  // The wildcard `*` automatically includes `d` and `sel` in the sensitivity list.
  always @(*) begin
    case (sel)
      2'b00: y = d[0];
      2'b01: y = d[1];
      2'b10: y = d[2];
      2'b11: y = d[3];
      default: y = 1'b0; // Default case is good practice to avoid latches
    endcase
  end
```

```verilog
// RTL style 4-to-1 mux
module mux_4_to_1_rtl_assign (
    input  wire [3:0] d,    // 4 data inputs
    input  wire [1:0] sel,  // 2 select lines
    output wire       y     // 1-bit output
);

  // The conditional operator is nested to form the multiplexer logic.
  // The MSB of 'sel' selects between the top and bottom 2-to-1 muxes.
  // The LSB of 'sel' selects the final input from the chosen pair.
  assign y = sel[1] ? (sel[0] ? d[3] : d[2]) : (sel[0] ? d[1] : d[0]);

endmodule
```


#### Synthesize and Implementation

Back to the breadboard analogy, the "compilation" flow is closer to figuring out the "optimal" resource usage and connection allocation. 

Logic synthesis is similar to software compilation in its fundamental goal of translating a high-level description into a low-level, executable one. However, the processes and the output are fundamentally different, reflecting their distinct purposes.

The aforementioned 4-to-1 mux would be sythesized to 

![4-to-1 Mux Schematic/Diagram](https://www.edn.com/wp-content/uploads/articles-articles-4-to-1-multiplexer-circuit-diagram-1387783580.jpg)

The exact gate level representation in this schematic/diagram isn't critical to our discussion. To get into HDL design, a great book can be found [here](http://103.203.175.90:81/fdScript/RootOfEBooks/E%20Book%20collection%20-%202024/ECE/Effective%20Coding%20with%20VHDL%20(Ricardo%20Jasinski)%20(z-lib.org).pdf)

The gist is, *the "code" or "description" will be translated to a physical setup*.


![FPGA design](https://allaboutfpga.com/wp-content/uploads/2014/04/fpga-design1.jpg)

Software Compilation process is relatively linear. It follows a predictable sequence of steps like lexical analysis, parsing, and code generation.

Logic Synthesis output is a ["netlist"](https://en.wikipedia.org/wiki/Netlist) a low-level description of logic gates and their interconnections, which is then used to physically lay out a circuit. The process is highly iterative and complex, often involving multiple passes. It is an optimization problem that balances multiple goals like minimizing area, power consumption, and meeting timing constraints. The Focuses on optimizing a physical artifact. Constraints include physical area, power usage, and propagation delay (timing). These are often conflicting goals. In real world, the error bound is often extremely narrow. 

Routing problems are generally, NP-complete or NP-hard, that is another can of worms, so we wouldn't further the discussion into [place & route](https://en.wikipedia.org/wiki/Place_and_route).

## GPU 

The core of GPU's parallel processing capability lies in [Streamming-Multiprocessors (SM)](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor)

*They are roughly analogues to the cores of CPUs. That is, SMs both execute computations and store state available for computation in registers, with associated caches. but no speculative execution*

There are two perspectives when describing CUDA's programming model, programmer's view and hardware perspective. 


<!-- From Hardware Perspective, if we use NVIDIA's terminology, each GPU  -->

<!-- If we use NVIDIA's terminology, each GPU has multiple (hundreds sometimes) SMs, [A SM is made up of 4 partitions](https://stackoverflow.com/questions/76638686/understanding-warp-scheduler-utilization-in-cuda-maximum-concurrent-warps-vs-re) (processing blocks), each of which having its own (warp) schedulers; and that warps  -->


<!-- Warp is the fundamental unit of execution, representing a bundle of 32 threads that are processed in a single clock cycle through SIMD fashion by the hardware.  -->


From a hardware perspective, The abstraction is mapped onto prefabricated the GPU's physical architecture. Unlike FPGA, we can't "redefined/reconnect" the implementation details of this architecture or the internal blocks. using NVIDIA's architecture as an example,  [an SM is made up of 4 partitions](https://stackoverflow.com/questions/76638686/understanding-warp-scheduler-utilization-in-cuda-maximum-concurrent-warps-vs-re)(processing blocks) and each partition has its own warp scheduler to manage the workload.

The core execution unit is the "warp," a group of 32 threads. All threads within a warp execute the same instruction simultaneously, following a Single Instruction, Multiple Data (SIMD) model ([where someone would argue the correct terminology should be SIMT](https://www.glick.cloud/blog/simt-vs-simd-parallelism-in-modern-processors)). The warp scheduler assigns warps to the SM's resources for execution.


### Scheduler 


![Terminal-GH100](https://modal-cdn.com/gpu-glossary/terminal-gh100-sm.svg)

Cache, memory and thread execution is handled by the runtime, though programmers might still want to manage memory and thread divergence through algorithm redesign and/or specific coding style to provide enough hints for compilers to reduce such phenomenon. It doesn't break the model that there are dedicated pre-fabricated hardware logic for handling instruction or data flow.


#### Queue


instructions are going through [queues along with registers and constants](https://forums.developer.nvidia.com/t/understanding-instruction-dispatching-in-volta-architecture/108896/5) 

There are other hardware details like [Load Store Unit(LSU)](https://modal.com/gpu-glossary/device-hardware/load-store-unit) that is worth a separate writeup, we wouldn't touch here.

*The take-away is together with scheduler, the system acting like central nerve system, it forms this "task storage and execution decision" combo at "instruction" and "data unit" level in hardware with the assistant of software (compiler) propagate instructions and data flows to execution units about what and when to do something*


#### Context Switching

GPUs have huge [register files](https://en.wikipedia.org/wiki/Register_file) comparing to CPUs, for example, an SM in the [NVIDIA Tesla V100 has 65536 (64K) x 4B registers](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels) to exploit thread level parallelism. 

One important aspect of optimization for CUDA performance is avoiding [register spilling](https://stackoverflow.com/questions/23876594/cuda-local-memory-register-spilling-overhead)



[*A large part of the thread's state is contained in its associated registers*](https://forums.developer.nvidia.com/t/why-in-thread-context-switching-there-is-no-need-to-store-state/38196) which helps GPU to perform context switch in a much faster fashion than classical CPUs. This could serve as a classical example of HW/SW co-design.

### Design Philosophy

*In processor design flexibility leads to complexity. The guiding principle of GPU design is minimizing the complexity of handling control flow and to a lesser extent, data access. This saves square millimeters on the die that can then be used for (1) more execution units, (2) a larger or smarter on-chip memory hierarchy, roughly in that order. For workloads that can benefit from massive parallelism, GPUs owe their performance advantage vs CPUs to focusing on these two aspects.*

*Note that the die sizes of the highest-performing CPUs and GPUs are close to the limit of what is manufacturable (currently around 850 square millimeters; a Xeon Platinum 9200 die is ~700mm^2, a H100 die is ~ 810 mm^2), so design trade-offs have to be made. One cannot “have it all”. Since larger die size translates to larger cost, these trade-offs similarly apply to lower-cost, lower-performing variants at various price points.*

*This leads to divergent design philosophies. CPUs are optimized for low latency and irregular control flow and data access patterns, with large on-chip memories and a decent number of execution units. GPUs are optimized for high throughput, regular control flow and data access patterns, with an extremely high number of execution resources and decent size on-chip memories. In the near future we will likely see tightly coupled CPU/GPU combos that reap the benefit of both worlds. One way to achieve this is to build processors from multiple dies (in a single package) which are sometimes called chiplets.*


[Practical reasons to share control unit amongst processing units](https://nichijou.co/cuda2-warp/)

Control units are quite complex:
  - Sophisticated logic for fetching instructions
  - Access ports to the instruction memory (Creating multi-port SRAM is challenging at the transistor level, read-write synchronization is always tricky)
  - on-chip instruction caches to reduce the latency of instruction fetch.

Having multiple processing units share a control unit can result in significant reduction in hardware manufacturing cost and power consumption.

### The Cost of Programmability

[Bill Daly (Chief Scientist of Nvidia) argues that](https://semiengineering.com/is-programmable-overhead-worth-the-cost/) 

 *The overhead of fetching and decoding, all the overhead of programming, of having a programmable engine, is on the order of 10% to 20% — small enough that there’s really no gain to a specialized accelerator. You get at best 20% more performance and lose all the advantages and flexibility that you get by having a programmable engine*


This could be the next topic we investigate like energy or other physical cost of "programmability".

[The cost of memory access is high](https://semiengineering.com/is-programmable-overhead-worth-the-cost/), 

These days, the only game in town seems about optimizing data movement (or not moving data as much as possible). This shift is notable because, while the dataflow approach has traditionally been considered complex, deep neural networks are inherently built on this principle.



<!-- During the research of this write-up, I stumbled upon [Michael Conrad](https://en.wikipedia.org/wiki/Michael_Conrad_(biologist))'s research on expressing such trade-off

*A computing system cannot at the same time have high programmability, high computational efficiency and high evolutionary adaptability.* -->


<!-- Dataflow design is often considered harder as it requires a different mindset ? -->

<!-- 
AI -> dataflow,
The cost of  -->


<!-- He has extensive researches and publication in theoretical biology and computer science. -->

TODO: History

GPUs are root from Vector and Array Processors and *Graphics Pipelines*

FPGA *programming* can broadly be seen as a form of concurrent programming with explicit memory management (without automatic CPU-style cache), more about CPU cache behavior can be found [here](./2024-10-31-CPU-Memory-Model-pt1.md)

- Lack of Automatic Cache: A key characteristic of traditional CPU programming is the complex, hardware-managed cache hierarchy that transparently handles data and instruction fetching to hid memory latency. FPGAs typically do not have this built-in, automated caching system. Memory access is explicit and deterministic:

  - Use on-chip memory blocks (Block RAMs or BRAMs) as scratchpads or specific, user-designed memory structures
  


## Notes

Co-design is for balancing control and programmability across a spectrum of granularity. We have things like ASICs, FPGAs, and GPUs people are also trying to get CGRA(Coarse Grain Reconfigurable Architecture) working.

### HLS

There are corpus research being done using traditional control flow languages like C/C++ to model or express such hardware mapping called [High Level Synthesize (HLS)](https://raw.githubusercontent.com/KastnerRG/pp4fpgas/gh-pages/main.pdf), personally, I find they are confusing 

They do contain the concept of "scheduling", but it is closer to static scheduling in the software world. 

### Warp Occupancy

The maximum number of concurrent warps per SM is limited and is likely different from architecture to architecture or from *Compute capability(CC)* to CC.

For example, [for Volta and Ampere Arch, this number is 64 or 48](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html). This number is also related to [*occupancy*](https://docs.modular.com/glossary/gpu/occupancy/)


<!-- I don't know if this is something like CAP theorem  -->



## Further reading

[Structure of an FPGA](https://digilent.com/blog/structure-of-an-fpga/)

[Tree Based Heterogeneous FPGA Architecture, Chapter 2](https://cse.usf.edu/~haozheng/teach/cda4253/)

[FPGA Memory Types](https://projectf.io/posts/fpga-memory-types/)

[A Tutorial on FPGA Routing Daniel Gomez-Prado Maciej Ciesielski](http://www.gstitt.ece.ufl.edu/courses/fall19/eel4720_5721/reading/Routing.pdf)

[The Price of Programmability](http://www0.cs.ucl.ac.uk/staff/W.Langdon/ftp/papers/Con88.pdf)

[Maximum number of warps and warp size per SM](https://forums.developer.nvidia.com/t/maximum-number-of-warps-and-warp-size-per-sm/234378/2)

[Invalidate+Compare: A Timer-Free GPU Cache Attach Primitive](https://github.com/0x5ec1ab/invalidate-compare/blob/main/attack-primitive/primitive-example.cu)

[Can L2 cache persistant policy be changed when kernel is running](https://forums.developer.nvidia.com/t/can-l2-cache-persistant-policy-be-changed-when-kernel-is-running/220816)

[Advanced Topics in Computer Architecture Lecture #8: GPU Warp Scheduling Research + DRAM Basics](https://pages.cs.wisc.edu/~sinclair/courses/cs758/fall2019/handouts/lecture/cs758-fall19-gpu_schedRes_dram.pdf)

<!-- [ML for ML Systems](https://courses.cs.washington.edu/courses/cse599m/23sp/) -->

<!-- TODO: PTX, concurrent algorithm and atomic hardware, cooperative parallelism -->
<!-- [Accelerated Computing Programming GPUs CSE599 I](https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf) -->

<!-- [How does the LSU load store unit execute load store instructions in the ampere architecture](https://forums.developer.nvidia.com/t/how-does-the-lsu-load-store-unit-execute-load-store-instructions-in-the-ampere-architecture/273699) -->

<!-- [Whats the difference between MIO and LSU instruction queue in Volta architecture](https://forums.developer.nvidia.com/t/whats-the-difference-between-mio-and-lsu-instruction-queue-in-volta-architecture/124749) -->

<!-- [From SIMD to SIMTs](https://www.youtube.com/watch?v=KCYlEub_8xc) -->

<!-- [When does MIO throttle stall happen](https://stackoverflow.com/questions/66233462/when-does-mio-throttle-stall-happen) -->