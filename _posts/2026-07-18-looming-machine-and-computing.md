---
layout: post
title: "Looming Machine and Computing"
categories: 
---

## Jacquard Machine

I stumbled upon [Donald & Era Farnsworth](https://www.magnoliaeditions.com/artists/donald-era-farnsworth/)'s stunning tapestry artwork's [River Reflection](https://www.magnoliaeditions.com/artworks/river-reflection/) in a local museum, [^3] and fell down a rabbit hole. 

Looking closely at the fabric, I realized every single thread has a perfectly clean color cut, meaning it wasn't painted on after the fact. It had to be done by a "digital loom". This led me to the origin of the [Jacquard Machine](https://en.wikipedia.org/wiki/Jacquard_machine) itself, named after [its inventor](https://en.wikipedia.org/wiki/Joseph_Marie_Jacquard), which dates back to [the high time of the first industrial revolution](https://spectrum.ieee.org/the-jacquard-loom-a-driver-of-the-industrial-revolution). It kicked off a big society shift that deserves a dedicated discussion. [^20] 

It was the very first machine to use replaceable [punch cards to control a sequence of operations](https://en.wikipedia.org/wiki/Jacquard_machine), considered a [landmark achievement in programmability](https://en.wikipedia.org/wiki/History_of_computing_hardware). 
The Jacquard loom is often considered one of the predecessor to modern computing because its interchangeable punch cards inspired the design of early computers. [^2] [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace) beautifully observed that

> The Analytical Engine weaves algebraic patterns, just as the Jacquard loom weaves flowers and leaves.
> --  Ada Lovelace, mathematician (1843) [^1]

The linked article is a great read that covering how the loom works and its direct connection with Charles Babbage's [Analytical engine](https://en.wikipedia.org/wiki/Analytical_engine).

If we look at old weaving manuals (like Ziegler from 1677), the grid notation is literally a primitive form of spatial (data) encoding. 

<p align="center">
    <img src="/images/weaving_pattern.png">
</p>

Page from the book Ziegler 1677[from Schneider 2007] [^9]

![Schematic illustration of the modern form of pattern notatation](https://figures.academia-assets.com/58765678/figure_005.jpg)
Schematic illustration of the modern form of pattern notation in relation to the parts of the shaft loom. Madelyn van der Hoogt, The Complete Book of Drafting for Handweavers [1993] [^10]

Precise Patterning: The jacquard automatic loom mechanism selects individual warp threads, creating intricate patterns as the weft is inserted.

## The Thread of Execution

It is no coincidence that our modern computing vocabulary, from "threads", "warps" to "pattern", is deeply rooted in the art of textiles.

The historical link between weaving and computing extends even deeper than the metaphors; it is embedded in our mental model of concurrency.

![Plain weave, with warp and weft labeled](https://cdn.sparkfun.com/assets/home_page_posts/6/4/1/1/350px-Warp_and_weft_2.jpeg)

While the Jacquard loom is not [Turing Complete](https://en.wikipedia.org/wiki/Turing_completeness), its design architecture conceptually resembles modern dedicated accelerators like [LPU](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/). It operators entirely on compiler-driven, static scheduling. Lacking dynamic control flow, caches, branch predictors, or out-of-order execution engines, the loom functions as a single, massive, [deterministic dataflow pipeline](https://en.wikipedia.org/wiki/Dataflow_architecture). Data "flows" through the hardware in rigid, lockstep fashion. For the loom, Mechanical gears move in perfect structural alignment, ensuring that individual warp and weft threads never collide or arrive out of sync.

This architecture reflects a pure dataflow pattern where the boundary between code and data [blurs](https://andr01d-1.github.io/misc/2024/04/22/code-as-data-data-as-code.html). The physical punch card acts simultaneously as the data structure (the pattern) and the execution instruction (which warp thread to lift). (Hmm....time for an installation art project?) [^19]

![Core Rope Memory](https://static.righto.com/images/agc-rope/core-rope-internal.jpg)

This intersection became literal reality during the space race with [core rope memory](https://en.wikipedia.org/wiki/Core_rope_memory). famously utilized in Apollo Guidance Computer. [Software physically becomes hardware](https://spectrum.ieee.org/software-as-hardware-apollos-rope-memory)[^12]. This high-density memory was [mostly handwoven by women](https://wehackthemoon.com/tech/core-rope-memory-when-computer-science-meets-girl-power).

For a long time, "programming" was widely considered ["Women's work"](https://news.sparkfun.com/6411). However, female contributions to computing were largely written out of the history until recently. The linked article is a great writeup covering this history, including how gender perceptions in the industry changed. It definitely merits a read. [^11]

Over two centuries after the digital automation born from weaving [^17], the downstream automation is once again causing profound societal and economic upheaval. 



## References:

[^1]: [Programming Patterns: The Story of The Jacquard Loom](https://www.scienceandindustrymuseum.org.uk/objects-and-stories/jacquard-loom)

[^2]: [Lovelace, Turing and the invention of computers](https://www.sciencemuseum.org.uk/objects-and-stories/lovelace-turing-and-invention-computers)

[^3]: Slightly more detailed description at [Magnolia Editions FAQ](https://www.magnoliaeditions.com/faq/)

[^4]: [The Surprisingly Political History of Knitting Machines](https://circularknittingjourney.substack.com/p/the-surprisingly-political-history)

[^5]: [Computerized modification](https://www.ayab-knitting.com/)

[^6]: Potential Sourcing 1: [PhotoWeavers](https://photoweavers.com/products/woven-throw)

[^7]: Potential Sourcing 2: [Prodigi](https://www.prodigi.com/) Claiming to be more eco-friendly

[^8]: [Data Visualization with Textiles](https://textiledataviz.org/) from Stanford

[^9]: [Weaving as Binary Art and the Algebra of Patterns](https://zenodo.org/records/3342554)

[^10]: [Programmed Images: Systems of Notation in Seventeenth- and Eighteenth-Century Weaving](https://www.academia.edu/38684097/Programmed_Images_Systems_of_Notation_in_Seventeenth_and_Eighteenth_Century_Weaving)

[^11]: ["Women's Work" and the Hidden History of Computer Science and Engineering](https://news.sparkfun.com/6411)

[^12]: [Software woven into wire: Core rope and the Apollo Guidance Computer](https://www.righto.com/2019/07/software-woven-into-wire-core-rope-and.html)

[^13]: Binary and Woven Art project [Fragmented Memory](https://phillipstearns.wordpress.com/fragmented-memory/) and [its store](https://glitchtextiles.com/woven-throws/dcp02802) 

[^14]: [Adding Texture to Tapestry](https://www.kennitatully.com/tapestry-journeys/adding-texture-to-tapestry)

[^15]: [Programming Patterns The Story of THe Jacquard Loom](https://www.scienceandindustrymuseum.org.uk/objects-and-stories/jacquard-loom)

[^16]: [Jacquard woven blankets: How to achieve the perfect finish](https://www.prodigi.com/blog/jacquard-woven-blankets-achieve-the-perfect-finish/)

[^17]: The power loom, is the defining icon of the First Industrial Revolution's textile boom, not Jacquard Machine.

[^18]: [AdaCAD](https://docs.adacad.org/docs/about/)

[^19]: Blankets as digital storage.....?

[^20]: A refreshing perspective, [Blood In the Machine](https://www.hachettebookgroup.com/titles/brian-merchant/blood-in-the-machine/9780316487740/)

[^21]: [Introduction to machine knitting](https://akaspar.pages.cba.mit.edu/machine-knitting/)