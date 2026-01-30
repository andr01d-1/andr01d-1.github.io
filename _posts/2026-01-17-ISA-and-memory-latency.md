---
layout: post
title: "ISA and Memory Latency"
categories: Micro-architecture
---

I have been working on "computers" for sometime, but I realize that **I don't know how a big part of computers work**. 

The last time I studied "assembly" consistently was a long time ago using a [8051 micro controller](https://en.wikipedia.org/wiki/Intel_MCS-51).

```assembly
; 8051 Assembly Program: Internal to External RAM Transfer
ORG 0000H           ; Start of program memory

    MOV R0, #30H    ; Load source pointer (Internal RAM start address)
    MOV DPTR, #2000H ; Load destination pointer (External RAM 16-bit address)
    MOV R2, #10     ; Initialize loop counter (transfer 10 bytes)

BACK:
    MOV A, @R0      ; Move data from Internal RAM (@R0) to Accumulator
    MOVX @DPTR, A   ; Move data from Accumulator to External RAM (@DPTR)
    
    INC R0          ; Increment source pointer
    INC DPTR        ; Increment destination pointer
    
    DJNZ R2, BACK   ; Decrement counter and repeat if not zero

    SJMP $          ; Halt program (infinite jump to current location)
END
```

For something like

```assembly
MOVX @DPTR , A
```

My mental model is, there would always be a "worst latency" cycle for these instructions.

Over the years, I dab into modern architecture assembly occasionally. But never put my mental model in test, until recently I started to think about how modern multi-core CPU works and how CPUs interface with things like memory.

An innocent looking modern X86 instruction

```assembly
mov r64, m64
```

shouldn't have many implications for a software engineer, right? 

Yeah, I have heard of [TLB](https://en.wikipedia.org/wiki/Translation_lookaside_buffer) and [MMU](https://en.wikipedia.org/wiki/Memory_management_unit), but my mental model is always that we can have a bounded access latency for this part of the system.

But that is incorrect. When [a TLB miss happening](https://stackoverflow.com/questions/37825859/cache-miss-a-tlb-miss-and-page-fault), MMU is forced to read from "RAM".

*When a [page fault](https://en.wikipedia.org/wiki/Page_fault) triggered, the issue is further aggravated, but it is out of the scope of this note*

## Reading from "Memory"

Does read/write from/to RAM have a fixed latency? It might have an upper bound for a specific hardware setup (thinking of RAM layout on specific board).

### How "memory" is physically connected to a "CPU"

This is a DDR4 SODIMM (Small Outline [Dual In-line Memory Module](https://en.wikipedia.org/wiki/DIMM)) layout from [Altium PCB layout guideline](https://resources.altium.com/p/pcb-routing-guidelines-ddr4-memory-devices)
<img src="https://files.resources.altium.com/sites/default/files/styles/max_width_1300/public/blogs/PCB%20Routing%20Guidelines%20for%20DDR4%20Memory%20Devices-73795.jpg?VersionId=eKa0QkGDLfJq4MAzxy5a6.q6mLW1TC64&itok=CAjU4AWy">

In general, for a modern CPU, the memory controller is typically [located within a SoC](https://www.design-reuse.com/article/60133-dram-controllers-for-system-designers/) 

**The signals travel through PCB traces, take time!!**

For a CPU based system, the DRAM calibration value are used entirely inside the CPU's memory controller/PHY to compensate for your PCB and package.

#### What the does with the value

- During boot, firmware (BIOS/UEFI or boot ROM) runs the controller's built-in training routines which measure your actual board delays and signal quality, then program those calibration registers automatically

- The integrated memory controller stores the calibration results (delay taps, impedance codes, Vref settings) in internal configuration registers or fuses and applies them to its I/O drivers and samplers for every memory access

#### [CAS latency](https://en.wikipedia.org/wiki/CAS_latency)

"[is how many clock cycles in it takes for the RAM module to access a specific set of data in one of its columns (hence the name) and make that data available on its **output pins**, starting from when a memory controller tells it to](https://www.tomshardware.com/reviews/cas-latency-ram-cl-timings-glossary-definition,6011.html)"


High-speed interfaces budgets this explicitly: timing analyses often constrain trace length to under about an inch or two so that PCB delay stays within the setup/hold margins defined by the DRAM timing.


Note: [Typical PCB propagation delay is on on the order of 150-180 ps per inc, so a few inches of routing only add about 0.3 ~ 0.5ns, versus ~10ns of CAS delay](https://www.protoexpress.com/blog/ddr4-vs-ddr5-the-best-ram/)


## WCET (Worst Case Execution Time)

Store/Load Buffers, cache hierarchy (within CPU and Multi-core SoC) already made instruction level latency non deterministic. 

More things like Buses, DRAM, prefetchers are source of such non-deterministic.

In most modern ISAs, the (worst) latency of memory instructions (like loads and stores) is intentionally left unspecified and determined.

The ISAs usually specifies what a load/store does (semantics, ordering rules, addressing models), but not how fast it must complete. Different cores with the same ISA can have different cache sizes, associativity, frequencies and memory controllers, so the cycle cost of a load that misses L1 but hits L2 or DRAM is inherently implementation dependent. Even within one core, **the same load instruction can vary in latency**


For a given x86 core, [vendor publishes tables of instruction *throughput* and *latency* assuming L1 cache hits](https://www.intel.com/content/www/us/en/content-details/679103/instruction-throughput-and-latency.html), but a load that misses in L1 and goes to L2, L3, or DRAM incurs large extra delays that are not encoded in the ISA itself.


Agner Fog's instruction tables list latencies/throughput for `mov` and many other x86 instructions, again per micro-architecture and with explicit caveat that cache misses and other effects can "increase the clock counts considerably"

- Performance models or hand-tuned code must be calibrated per micro-architecture (e.g., by micro-benchmark) rather than assuming an ISA-level guarantee.
## When the ISA Does Get Involved

The ISA does provide "hints" to help the hardware manage it

- Prefetch Instructions: Commands like `PREFETCH0` on X86 allow the programmer to tell the CPU, "a high-latency DRAM ops might be needed"

- Memory Barriers: Instructions that ensure all previous memory loads are finished before moving on, which is critical for multi-core synchronization

### Compilers 

TODO: Memory models


## Mission Critical SoC Design

It looks like for these types of systems, things like load/store buffers are still allowed?

This introduces challenges that must be addressed for certification:

- Predictability (WCET): For "Hard Real-Time" systems, the Worst-Case Execution Time (WCET) must be guaranteed. Buffers can introduce timing variability; for example, if a buffer is full, the CPU may suddenly stall, making timing analysis more complex.
    
- Data Integrity: Buffers can lead to "stale data" issues in multi-core systems if not properly managed by coherence protocols. Mission-critical designs often use hardware-managed coherence or strict software memory barriers to ensure the buffer is drained before critical shared-data access.

- Non-Deterministic Behavior: In some safety-critical modes, features like load-store forwarding (where a load reads directly from a pending store in the buffer) might be restricted or strictly analyzed to ensure they don't introduce race conditions

### Mitigation

- Infineon AURIX (TC1.6): Implements store buffering by default to improve performance but allows software to prioritize certain operations or [drain the buffer](https://community.infineon.com/t5/Knowledge-Base-Articles/How-does-the-TriCore-store-buffer-work-AURIX-MCU/ta-p/802402#.) for atomic accesses


Q: What about using general processor?

Q: How would software help?

Instruction reordering restricted

3. Compiler-Level Ordering and Safety

4. Hardware-Software Co-Design

Things like GPUArmor??/

## References

[University of Cambridge: 	 System-on-Chip Design and Modelling (Part II)
2009–10](https://www.cl.cam.ac.uk/teaching/0910/SysOnChip/sp6busnoc/zhp840f6cfcf.html)

[Titanium DDR DRAM PCB Design User Guide](https://www.efinixinc.com/docs/pcb-guidelines-ddr-ti-ug-v1.2.pdf)

[DDR4 initialization and calibration](https://www.systemverilog.io/design/ddr4-initialization-and-calibration/)

[Application Note AM62x, AM62Lx DDR Board Design and Layout
Guidelines](https://www.ti.com/lit/an/sprad06c/sprad06c.pdf)

[i.MX53 System Development User’s Guide](https://www.nxp.com/docs/en/user-guide/MX53UG.pdf)

[i.MX53 DDR Calibration](https://www.nxp.com/docs/en/application-note/AN4466.pdf)

[SDR SDRAM PCB Timing Budget](https://electrical.codidact.com/posts/288164)

[ARM or x86? ISA Doesn’t Matter](https://chipsandcheese.com/p/arm-or-x86-isa-doesnt-matter)

[Instruction tables By Agner Fog](https://www.agner.org/optimize/instruction_tables.pdf)

[Understanding CPU limitations with memory](https://www.crucial.com/support/articles-faq-memory/understanding-cpu-limitations-with-memory)

[What mechanism does CPU use to know if a write to RAM was completed?](https://electronics.stackexchange.com/questions/669470/what-mechanism-does-cpu-use-to-know-if-a-write-to-ram-was-completed)

[Designing a RISC-V CPU in VHDL, Part 17: DDR3 Memory Controller, Clock domain crossing](https://domipheus.com/blog/designing-a-risc-v-cpu-in-vhdl-part-17-ddr3-memory-controller-clock-domain-crossing/)

[Intel CPU die topology](https://jprahman.substack.com/p/intel-cpu-die-topology)

[Understanding Latency Hiding on GPU](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf)


Is this still applicable? 

[Architectural Principles for Safety-Critical Real-Time Applications](https://www.cs.unc.edu/~anderson/teach/comp790/papers/safety_critical_arch.pdf)

[MULTICORE CONSIDERATIONS FOR SAFETY-CRITICAL SOFTWARE APPLICATIONS](https://apps.dtic.mil/sti/trecms/pdf/AD1123249.pdf)

[Paper: Memory Barriers, A hardware View for Software Hackers](https://dannas.name/2020/06/19/memory-barriers)