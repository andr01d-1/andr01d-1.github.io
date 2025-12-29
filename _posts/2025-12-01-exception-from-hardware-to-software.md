---
layout: post
title: "Exception from Hardware to Software"
categories: Safety Hardware Software
---

This note might stay in WIP state for a long time. It is an attempt for taking a generic look at *Exceptions* and *Exception Handling*, instead of traditional hardware vs software division.

```
What hardware interrupts, software interrupts and exceptions all have in common is that they execute 
in different processor context that normal code
- typically switching to an independent stack and automatically pushing registers (or using an alternate register bank). 
They all operate via a vector table, and you cannot pass or return parameters via formal function parameter passing and return
```

I will try making distinctions between hardware and software exception when developing this note. 
The end goal is having a deeper understanding of (using) exceptions in embedded or mission critical systems, how it affects hardware and software design choices.

## Hardware vs Software Exceptions

```
A CPU/FPU exception will be caught by the kernel, converted
into a signal and delivered to the offending process. 
So for CPU exceptions, exception handling is effectively signal handling 
(e.g. handling SIGFPE for div by 0). 
Mapping CPU exceptions to C++ exceptions is not trivial,
since signals are fully asynchronous - which C++ exceptions are not
```

The article in Further Reading section has a bit more expanded content on hardware/SoC design for safety critical applications


Exception handling is a core part of [Application Binary Interface (ABI)](https://stackoverflow.com/questions/2171177/what-is-an-application-binary-interface-abi)

The specifications are often platform or architecture-specific, such as the Itanium C++ ABI used on many Linux platforms or the specific conventions for the [ARM Embedded ABI (EABI)](https://github.com/ARM-software/abi-aa) and [standards for 64-bit (AArch64)](https://github.com/ARM-software/abi-aa/tree/main/sysvabi64)

The ARM ABI defines architecture-specific conventions (like register usage and calling sequences) for the ARM processor family, while the [Itanium C++ ABI](https://itanium-cxx-abi.github.io) is more high level, language-specific conventions that ensure binary compatibility for C++


Though Itanium C++ ABI is designed for the now-defunct [Itanium architecture](https://en.wikipedia.org/wiki/Itanium) [IA-64](https://en.wikipedia.org/wiki/IA-64), this ABI became the de facto standard for C++ compilers (like GCC and Clang) across most Unix-like environments , including Linux, macOS, BSDs and QNX. Its primary focus is on C++ language features (e.g., *vtable* layout, *RTTI*, name mangling, exception handling...), which are then layered on top of the underlying platform's specific C-level ABI.

The ABI specifies key mechanisms like stack unwindling, the format of exception tables (metadata used by the runtime), and language-specific "personality routines" that handle language-specific tasks during the process


When reading about exceptions, we may encounter [Structured Exception Handling (SEH)](https://learn.microsoft.com/en-us/cpp/cpp/structured-exception-handling-c-cpp?view=msvc-170) is specific to Windows, other operating systems use different mechanisms for similar tasks; for example, Linux uses POSIX signals(like SIGSEGV for access violations) instead of SEH


<!-- ## Cache control  -->

## Compiler

[Fangrui](https://maskray.me), a [LLVM contributor](https://llvm.org/docs/ExceptionHandling.html) has a very detailed explanation of [stack unwinding](https://maskray.me/blog/2020-11-08-stack-unwinding) and [C++ exception handling ABI](https://maskray.me/blog/2020-12-12-c++-exception-handling-abi)

I will come back for a deep dive after going through Fangrui's blogs. Maybe expanding the idea of customized exception handling mechanism for smaller embedded system's codesize.

<!-- How QNX  -->


## Side Effects of (Software) Exceptions

[What happens if throw fails to allocate memory for exception object?](https://stackoverflow.com/questions/45497684/what-happens-if-throw-fails-to-allocate-memory-for-exception-object)

Throwing an exception almost always involves memory allocation, typically for the exception object itself, which can be on the stack or heap depending on the language/compiler, plus any data within that object (like strings).

The compilers might try to optimize it, often using custom runtime allocations rather than global `new/malloc`. If we are throwing built-in types or simple classes, the system handles allocation; if throwing complex objects, their constructors might perform further allocations, which can fail, leading to a failure-to-throw scenario if memory is critical low.

From QNX's [Heap Analysis: Making Memory Errors a Thing of the Past](https://www.qnx.com/developers/docs/6.3.2/neutrino/prog/hat.html) we could take a peak at how QNX handles heap related memory errors. A newer version of the doc can be found in reference

Dynamic Allocation after the startup phase in mission critical system is in general frowned upon, if not strictly prohibited. Hence, there are efforts to build [static exception mechanisms](https://www.eelco.de/posts/static-exceptions/) 





## Automotive and Other Mission Critical System Standards


```
ISO 26262

IEC 61508
```


Processor (functional) safety concepts

[Dual Core Lock-step](https://codasip.com/glossary/dual-core-lockstep/)

```
Two identical processor cores, or CPUs, operate in parallel, executing the same set of instructions simultaneously. The key feature of dual-core
lockstep is that both cores execute the same instructions and compare their results at every step to ensure they match.
```


Safety Mechanisms

```
    - Error Correcting Codes (ECC)

    - Built In Self Tests (BIST)
```


## Reference

### Hardware Exceptions

[Understanding hardwre interrupts and exceptions at processor and hardware level](https://stackoverflow.com/questions/33501438/understanding-hardware-interrupts-and-exceptions-at-processor-and-hardware-level)

[Servicing an interrupt vs servicing an exeption](https://stackoverflow.com/questions/48665196/servicing-an-interrupt-vs-servicing-an-exception)

[Cache Miss, TLB miss and Page fault](https://stackoverflow.com/questions/37825859/cache-miss-a-tlb-miss-and-page-fault)

[Single-cycle MIPS-based CPU design with interrupt and exception by Ya Min Li Chapter 6](https://www.edaplayground.com/x/5YT9)

[Computer Principles And Design In Verilog HDL](https://www.amazon.com/Computer-Principles-Design-Verilog-HDL/dp/1118841093)

[Book's codes and figure](https://www.wiley.com/legacy/wileychi/yamin/index.html?type=Home)

[MIPS32_PipelineCPU](https://github.com/kentang-mit/MIPS32_PipelineCPU)

[Always use feenableexcept() when doing floating point math](https://berthub.eu/articles/posts/always-do-this-floating-point/)

[Differences between System V ABI and C++ Itanium ABI](https://stackoverflow.com/questions/77441978/differences-between-system-v-abi-and-c-itanium-abi)

### QNX

[QNX programming guide: Heap analysis](https://www.qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.prog/topic/hat.html)

[FPU exception in QNX](https://www.qnx.com/support/knowledgebase.html?id=50130000000PhmA)

[QNX 8 doc: Rescheduling Exceptions](https://www.qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.getting_started/topic/s1_procs_Rescheduling_exceptions.html)

### Float Point

[Example of hardware and software support differences for floating-point arithmetic](https://developer.arm.com/documentation/dui0472/h/compiler-coding-practices/example-of-hardware-and-software-support-differences-for-floating-point-arithmetic)

#### GPU

[Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point/index.html)

[CUDA C++ Floating-Point Exceptions](https://www.aussieai.com/blog/cuda-floating-point-exceptions)

## Further Reading


### Processor and SoC

[The flexible approach to adding Functional Safety to a CPU](https://developer.arm.com/community/arm-community-blogs/b/embedded-and-microcontrollers-blog/posts/flexible-approach-to-adding-functional-safety-to-a-cpu)

[Designer’s Guide: Safety-critical processors](https://www.edn.com/designers-guide-safety-critical-processors/)

[A Practical guide to ARM Cortex-M Exception Handling](https://interrupt.memfault.com/blog/arm-cortex-m-exceptions-and-nvic)

[How can the L1, L2, L3 CPU caches be turned off on modern x86/amd64 chips](https://stackoverflow.com/questions/48360238/how-can-the-l1-l2-l3-cpu-caches-be-turned-off-on-modern-x86-amd64-chips)


### Compiler

[Safety-critical software and optimising compilers](https://softwareengineering.stackexchange.com/questions/267277/safety-critical-software-and-optimising-compilers)


### Software

[C++ links: error handling](https://github.com/MattPD/cpplinks/blob/master/error_handling.md)

[The true cost of C++ exceptions](https://mmomtchev.medium.com/the-true-cost-of-c-exceptions-7be7614b5d84)

Advices for general software exception handling

[C++ Exceptions and Memory Allocation Failure](https://yongweiwu.wordpress.com/2023/08/17/cxx-exceptions-and-memory-allocation-failure/comment-page-1/)