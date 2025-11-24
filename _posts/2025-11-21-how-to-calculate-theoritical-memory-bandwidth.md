---
layout: post
title: "How to Calculate Theoritical Memory Bandwidth"
categories: GPU, Programming Models, Chip Design
---

The note content is inspired by an enlightening converation with [Faradawn](https://faradawn.github.io/)

## NVidia Ampere

<p align="center">
    <img src="https://cdn.mos.cms.futurecdn.net/WriSXYzQsqxcnYcMpA2BfK-801-80.jpg.webp" />
</p>

[GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)'s 32 bit G6X memory interfaces connect to corresponding [off-chip](https://forums.developer.nvidia.com/t/how-to-correctly-understand-cuda-global-memory-v-s-off-chip-pysical-location/275722) on a RTX 3090 PCB [Micron GDDR6X](https://www.micron.com/about/blog/company/partners/latest-nvidia-gpus-powered-by-innovation-collaboration-micron-gddr6x) VRAM

<p align="center">
    <img src="/images/ampere_offchip_memory.png">
</p>

## Apple M Series

<p align="center">
    <img src="https://cdn.mos.cms.futurecdn.net/A3QSjPxt4oYcYuy8i6kRtX.jpg">
</p>

<p align="center">
    <img src="https://www.techspot.com/images2/news/bigimage/2023/05/2023-05-15-image-12-j_1100.webp">
</p>

<!-- Need to verify -->

<!-- For Apple M-series chips, the main SoC (System on a Chip) and the DRAM memory chips are mounted together in a single physical package, known as a
System-in-a-Package (SiP) or Multi-Chip Module (MCM) design

- Standard M1, M2, M3 chips (non-Pro/Max): These chips do not use a silicon interposer to connect the memory to the main SoC. Instead, the DRAM is typically placed on the same organic substrate as the main chip and uses a standard packaging technique (often referred to as "memory on package" or MoP). The memory and other components are all integrated into one custom package.

- M1, M2, M3 Ultra chips: These larger, high-performance chips are formed by connecting two Max-version dies using Apple's proprietary UltraFusion technology, which employs a passive silicon interposer. This interposer serves as a high-speed bridge between the two dies, creating a single, software-agnostic processor with a massive memory bandwidth and a large, unified memory pool. The memory chips themselves are still mounted on the package substrate, connected through traces that leverage the interposer's architecture for high bandwidth to the combined dies -->


<!-- Ultra System on Chip (SOC) is built using a  -->

<!-- Ultra Series leverage [UltraFusion Chip-to-Chip interconnect](https://www.tomshardware.com/news/tsmc-clarifies-apple-ultrafusion-chip-to-chip-interconnect) ([passive silicon bridge](https://past.date-conference.com/proceedings-archive/2023/DATA/1089.pdf) that connects one M1 Max to another M1 Max processor to build an M1 Ultra) -->


## DGX Spark's Theoritical Memory Bandwidth Calculation

- Memory Type and Speed

    LPDDR5x sppeds range from [6400 MT/S to 8533 MT/s](https://www.tomshardware.com/news/jedec-publishes-lpddr5x-specification)

- Memory Bus Width

    [16-channel 32 bit, a total bus width of 256 bits](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

The general formula for calculating the maximum theoritical memory bandwidth in Bytes per second is

$$\text{Bandwidth}=\frac{\text{Data Rate (MT/s)}\times \text{Bus Width (bits)}}{\text{Bits per Byte (8)}}$$

Plugging in the DGX Spark's values: 

$$\text{Bandwidth}=\frac{8533\,\text{MT/s}\times 256 \text{bits}}{8\,\text{bits/Byte}}$$

$$\text{Bandwidth}=8533\,\text{MByte/s}\times 32$$

$$\text{Bandwidth}\approx 273,056\,\text{MByte/s}$$





## References


[Infrared PhotoGrapher Photos Nvidia GA102 Ampere Silicon](https://www.tomshardware.com/news/infrared-photographer-photos-nvidia-ga102-ampere-silicon)

[Annotated Apple M3 Processor Die Shots Bring Chip Designs to Life](https://www.tomshardware.com/news/annotated-apple-m3-processor-die-shots-bring-chip-designs-to-life)

[Apple M3, M3 Pro & M3 Max - Chip Analysis](https://www.youtube.com/watch?v=8bf3ORrE5hQ&t=40s)

[GPU PCB Breakdown: RTX 3090 Kingpin Edition](https://www.youtube.com/watch?v=4gpo3TewxV8)

[Packaging Part 3 - Silicon Interposer](https://www.youtube.com/watch?v=WMQtD4hDHak&t=32s)