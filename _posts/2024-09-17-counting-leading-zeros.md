---
layout: post
title: "Counting leading zeros from software to hardware"
categories: low-level-GPU
---

Counting leading zeros/[finding first set](https://en.wikipedia.org/wiki/Leading_zero) is a widely used in [DSP normalization, finding the highest priority task in schedulers, and implementing integer logarithm functions](https://www.state-machine.com/fast-deterministic-and-portable-clz).

I recently encountered below naive implementation in one project

```c++
#include <cstdint>

constexpr uint32_t countLeadingZero(uint32_t value) noexcept
{
   if (value == 0u)
   {
       return 32U;
   }
   uint32_t count{0u};
   for (uint32_t i{32u}; i >= 1u; --i)
   {
       uint32_t mask{1u << (i - 1u)};
       if ((value & mask) == 0u)
       {
           count = 33u - i;
       }
       else
       {
           break;
       }
   }
   return count;
}

int main()
{
   uint32_t what{0x101232};
   return countLeadingZero(what);
}
```

As we can see, a loop is in generated assembly using x86-64 gcc 14.2 `-O3`


```asm
main:
        mov     eax, 32
        mov     esi, 1
.L2:
        mov     edx, eax
        sub     eax, 1
        je      .L9
        lea     ecx, [rdx-2]
        mov     edi, esi
        sal     edi, cl
        mov     ecx, edi
        and     ecx, 1053234
        je      .L2
.L4:
        mov     eax, 33
        sub     eax, edx
        ret
.L9:
        mov     edx, 1
        jmp     .L4
```

[The fastest portable approaches to simulate clz are a combination of binary search and table lookup](https://en.wikipedia.org/wiki/Leading_zero)

The discussion about branch and pipeline on the same page is interesting.

An algorithm similar to de Bruijn multiplication for CTZ works for CLZ,

The algorithm has a fixed execution cycle. (branchless) and easier for constexpr evaluation 

```c++
constexpr uint32_t countLeadingZero(uint32_t value) noexcept
{
   constexpr uint8_t DeBruijnLookup[32] = {
       31, 22, 30, 21, 18, 10, 29, 2, 20, 17, 15, 13, 9, 6, 28, 1,
       23, 19, 11, 3, 16, 14, 7, 24, 12, 4, 8, 25, 5, 26, 27, 0
   };


   if (value == 0) return 32;


   value |= value >> 1;
   value |= value >> 2;
   value |= value >> 4;
   value |= value >> 8;
   value |= value >> 16;


   return DeBruijnLookup[(uint32_t)(value * 0x07C4ACDDU) >> 27];
}
```

## Compiler-Specific Intrinsics

These intrinsics are typically implemented using hardware-specific instructions, making them very efficient. 

For GCC and Clang, 

- __builtin_clz for unsigned int
- __builtin_clzl for unsigned long
- __builtin_clzll for unsigned long long

### X86 Architecture

On x86 processors with LZCNT instruction (available in some newer Intel and AMD CPUs)

```asm
LZCNT EAX, EBX  ; Count leading zeros of value in EBX, store result in EAX
```

If LZCNT is not available, we can use the `BSR (Bit Scan Reverse` instruction with some additional logic:
text

```asm
BSR ECX, EAX    ; Find index of highest set bit
XOR ECX, 31     ; Invert to get leading zero count
MOV EAX, 32     ; Handle special case for input of 0
CMOVNE EAX, ECX ; If input wasn't 0, use the calculated value
```

on ARM architecture, the intrinsics might reduce to  [CLZ](https://developer.arm.com/documentation/dui0491/i/Compiler-specific-Features/--clz-intrinsic)

Instruction on modern ARM processors typically takes one cycle for [such operation](https://hardwarebug.org/2014/05/15/cortex-a7-instruction-cycle-timings/)


**What about issue and result latency?**
One cycle latency is the ideal case when there are no stalls or dependencies

**We should still use compiler instrinsics for portability**

Let's take one step further, how would we implement such operation in HDL.

```verilog
module clz_32bit(
    input [31:0] a,
    output reg [5:0] count
);

    always @(*) begin
        casez(a)
            32'b1???????????????????????????????: count = 6'd0;
            32'b01??????????????????????????????: count = 6'd1;
            32'b001?????????????????????????????: count = 6'd2;
            32'b0001????????????????????????????: count = 6'd3;
            32'b00001???????????????????????????: count = 6'd4;
            32'b000001??????????????????????????: count = 6'd5;
            32'b0000001?????????????????????????: count = 6'd6;
            32'b00000001????????????????????????: count = 6'd7;
            32'b000000001???????????????????????: count = 6'd8;
            32'b0000000001??????????????????????: count = 6'd9;
            32'b00000000001?????????????????????: count = 6'd10;
            32'b000000000001????????????????????: count = 6'd11;
            32'b0000000000001???????????????????: count = 6'd12;
            32'b00000000000001??????????????????: count = 6'd13;
            32'b000000000000001?????????????????: count = 6'd14;
            32'b0000000000000001????????????????: count = 6'd15;
            32'b00000000000000001???????????????: count = 6'd16;
            32'b000000000000000001??????????????: count = 6'd17;
            32'b0000000000000000001?????????????: count = 6'd18;
            32'b00000000000000000001????????????: count = 6'd19;
            32'b000000000000000000001???????????: count = 6'd20;
            32'b0000000000000000000001??????????: count = 6'd21;
            32'b00000000000000000000001?????????: count = 6'd22;
            32'b000000000000000000000001????????: count = 6'd23;
            32'b0000000000000000000000001???????: count = 6'd24;
            32'b00000000000000000000000001??????: count = 6'd25;
            32'b000000000000000000000000001?????: count = 6'd26;
            32'b0000000000000000000000000001????: count = 6'd27;
            32'b00000000000000000000000000001???: count = 6'd28;
            32'b000000000000000000000000000001??: count = 6'd29;
            32'b0000000000000000000000000000001?: count = 6'd30;
            32'b00000000000000000000000000000001: count = 6'd31;
            32'b00000000000000000000000000000000: count = 6'd32;
        endcase
    end

endmodule
```

This implementation is fully combinatorial and should synthesize to efficient hardware on most FPGA or ASICs.

I would guess this is the reason the single clock cycle operation is limited to 32 bit or 64 bit only implmentation?


[Area vs. Speed Tradeoff: Different CLZ implementations offer tradeoffs between area usage and speed. For example: The cascaded mux approach (like the one shown in the Verilog code earlier) is relatively compact but may have longer propagation delays. A hierarchical approach (building larger CLZ units from smaller ones) may use more area but could offer lower propagation delays](https://stackoverflow.com/questions/2368680/count-leading-zero-in-single-cycle-datapath)

Would scalability be a concern? 

Power Consumption: The combinational nature of CLZ means it could potentially have high dynamic power consumption if the input changes frequently. Power gating or other power optimization techniques might be considered for low-power designs.

## How is CLZ related to FPU (Floating Point Unit) design 

Normalization: CLZ is crucial for normalizing floating point numbers. After arithmetic operations, the result may need to be normalized by shifting the mantissa and adjusting the exponent. CLZ helps determine how many shifts are needed. 

This is [how](https://github.com/BrunoLevy/learn-fpga/blob/61b9b1c18d3962f4c2c2d0b57356c531f69b9424/FemtoRV/RTL/PROCESSOR/TESTDRIVE/femtorv32_testdrive_RV32IM_simF.v#L34) it is [done](https://github.com/BrunoLevy/learn-fpga/blob/master/FemtoRV/TUTORIALS/FPU.md) in emulated RV32 CPUs

## Safety:

Some implementations may cause errors when input is 0. Might make sense to do an input guarding check. 

For example:

CUDA also provides [intrinsic device functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html)

```c++
__device__ int __clz(int x)

__device__ int __clzll(long long int x)
```

