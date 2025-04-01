---
layout: post
title: "Poke A Hole"
categories: tooling
---

I was recently tasked with optimizing startup memory usage for a mission-critical system. To understand the existing memory usage of a particular class design, my first instinct was to use GDB to load the executable with debug symbols enabled and then run:

```bash
ptype /o class_or_struct_name
```

for [examing the symbol table](https://doc.ecoscentric.com/gnutools/doc/gdb/Symbols.html)

Even though we could automate this process using a [gdb init file](https://cgi.cse.unsw.edu.au/~learn/debugging/modules/gdb_init_file/), it still felt suboptimal. A co-worker suggested that we might be able to directly inspect the (shared object) `so` file without requiring runtime invocation.

If we run

```bash
objdump -d -C -g lib_name.so 
```
to inspect the binary, we would notice _DWARF_ tags like [`DW_AT_name` and others](https://www.ibm.com/docs/en/zos/2.5.0?topic=entries-source-file-name).

[DWARF](https://en.wikipedia.org/wiki/DWARF) shorts for _Debugging With Attribute Record Format_. It is used by [many compilers and debuggers to support source level debugging](https://dwarfstd.org/).

It turns out we could use a handy [open source tool](https://github.com/acmel/dwarves) called [Pahole](https://www.mankier.com/1/pahole) (Poke-A-Hole)

```
$ pahole list_head
struct list_head {
	struct list_head *         next;                 /*     0     8 */
	struct list_head *         prev;                 /*     8     8 */

	/* size: 16, cachelines: 1, members: 2 */
	/* last cacheline: 16 bytes */
};
$
```

It also supports other metadata formats designed for encoduing debug information like [BTF (BPF Type Format)](https://docs.kernel.org/bpf/btf.html) and [CTF (Compact C Type Format)](https://lwn.net/Articles/795384/).

Here is a [great read](https://novitoll.com/posts/2024-4-21/pahole.html) on how the author uses Pahole to optimize struct packing across multiple languages (C, Go, Rust)

<!-- TODO: BPF's ring buffer for multi-CPU event tracing -->
<!-- https://github.com/torvalds/linux/blob/master/kernel/bpf/ringbuf.c -->

Yes, just like `structs`, the sequence of class members in C++ does affect memory packing and overall memory layout. The C++ standard guarantees that the members of a class or struct appear [in memory in the same order as they are declared](https://jonasdevlieghere.com/post/order-your-members/)

[In multi-threaded scenarios, the layout of atomic members can impact performance. Placing atomic members on separate cache lines can prevent false sharing and improve parallel execution](https://stackoverflow.com/questions/892767/optimizing-member-variable-order-in-c)

Alignment also has implications for SIMD (Signle Instruction, Multiple Data) operations.

### Improvement

Pahole can also suggest [improvements](https://haryachyy.wordpress.com/2018/06/15/learning-optimization-structures-analysis-with-pahole-tool/) 

```
pahole --show_reorg_steps --reorganize -C structure_name a.out
```


### Further Reading:

[The Lost Art of Structure Packing](http://www.catb.org/esr/structure-packing/)

[Hello eBPF auto layouting structs](https://mostlynerdless.de/blog/2024/03/25/hello-ebpf-auto-layouting-structs-7/)

[10.39M Storage I/O Per Second From One Thread](https://spdk.io/news/2019/05/06/nvme/)

### Post Mortem

Google's open source size profiler for binaries [bloaty](https://github.com/google/bloaty)