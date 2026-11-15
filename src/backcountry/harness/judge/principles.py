"""Why kernels are fast, slow, exact or not on Apple GPUs under MLX, for the
judge, read on request through techniques_reference.

Every entry was checked against its sources on 2026-10-16 (Apple's Metal
documentation and talks, MLX 0.32.4's source, reverse-engineering and
community work); wording a check could not support was cut or corrected.
Numbers this tool measured itself say so. MLX details are for the installed
version and can change with it. Entries: (topic, principle, sources).
"""

PRINCIPLES = (
    # -- the hardware --------------------------------------------------------
    ("hardware", "Each threadgroup runs on one GPU core and uses that core's threadgroup memory. "
     "Size a dispatch by threads, not threadgroups: Apple's rule of thumb is about 1K-2K "
     "concurrent threads per core, across all cores (chip.gpu_cores). Smaller threadgroups, "
     "a multiple of 32, let the GPU balance work better. A tiny dispatch pays ramp-up and "
     "tail time no matter how good its code is.",
     "Apple WWDC22 'Scale compute workloads across Apple GPUs' (10159)"),
    ("hardware", "A SIMD group is 32 threads on every Apple GPU reported so far. Only threads in "
     "one SIMD group are guaranteed to run at the same time: never spin-wait on another SIMD "
     "group or threadgroup, it can deadlock. Lanes that branch differently run both paths "
     "under a mask. SIMD-group matrix operations and other SIMD collectives must run under "
     "control flow that is the same for the whole SIMD group, or the behavior is undefined.",
     "Apple WWDC22 10159; Apple engineer, developer.apple.com/forums/thread/762656; Metal "
     "Shading Language spec 4.1 section 2.4; Dougall Johnson's Apple G13 GPU docs"),
    ("hardware", "A threadgroup can use at most 32 KB of threadgroup memory on M1 through M5. "
     "1,024 threads per threadgroup is a theoretical maximum; a pipeline's "
     "maxTotalThreadsPerThreadgroup can be lower when it uses many registers.",
     "Apple Metal Feature Set Tables (Apple7-Apple10 columns, footnote 5)"),
    ("hardware", "Occupancy (SIMD groups resident per core) hides memory latency. Before M3, a SIMD "
     "group reserved its peak register count for its whole life, so a register-hungry "
     "kernel ran fewer groups. On M3 and later, registers, threadgroup memory and caches "
     "share each core's on-chip memory and are allocated by actual use, and the hardware may "
     "cap occupancy to avoid cache thrashing. So a tile's occupancy cost must be measured, "
     "not computed, and more resident groups is not always faster.",
     "Apple tech talks 'Explore GPU advancements in M3 and A17 Pro' (111375) and 'Discover "
     "new Metal profiling tools for M3 and A17 Pro' (111374)"),
    ("hardware", "Matrix hardware: M1-M4 have simdgroup_matrix, 8x8 multiply-accumulate on half "
     "and float (bfloat needs Metal 3.1 and is not documented as native on every chip). "
     "Apple documents no dedicated matrix unit before M5, and measured throughput fits it "
     "running on the ordinary ALUs. M5 adds a Neural Accelerator in each GPU core, reached "
     "through Metal 4 tensor operations; stock MLX uses it on M5, so a hand kernel without "
     "it can lose badly there.",
     "Apple Metal Feature Set Tables; MSL spec 4.1 section 2.4; Apple Machine Learning "
     "Research 'Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU'; "
     "github.com/philipturner/metal-benchmarks; oMLX source comment (deepseek_v4/switch_layers.py)"),
    ("hardware", "Half (FP16) arithmetic runs at the same peak rate as float on M1 and later, not "
     "twice it, but uses fewer registers. Apple does not document which GPUs do bfloat "
     "arithmetic in hardware, so do what MLX does: load bfloat, compute in float, round once "
     "where the original rounds. Never change the model's precision.",
     "github.com/philipturner/metal-benchmarks (FMA per core-cycle); Apple tech talk 111375"),
    ("hardware", "Threadgroup memory is banked, but Apple publishes no bank count, and the "
     "penalty pattern differs by generation (M3 behaves unlike M1). Consecutive 4- to 16-byte "
     "elements per lane is the safe default; measure a strided pattern rather than assume "
     "NVIDIA-style 32-way conflicts. This tool measured a SIMD group reading floats 32 apart "
     "at 4.6x the time of consecutive ones on an M4.",
     "community microbenchmark (techboards.net, 'The mystery of Apple M3 on-chip shared "
     "memory'); Apple GPU engineer's advice relayed on the raywenderlich forums; this tool's "
     "M4 measurement"),
    ("hardware", "CPU and GPU share one DRAM, so there is no copy, but the bandwidth is shared "
     "and the published figure is a peak: M4 120 GB/s, M4 Pro 273, M4 Max 410 or 546, M5 153. "
     "A kernel achieves less; plan against chip.bandwidth_gbps, measured on this Mac.",
     "Apple newsroom and specs pages; Apple Machine Learning Research M5 post"),

    # -- memory traffic, latency and launches ----------------------------------
    ("memory", "Lay out a SIMD group's device-memory loads so that together they cover "
     "contiguous, aligned cache lines (128-byte lines are reported). Strided or scattered "
     "loads drag in whole lines that are mostly unused, wasting bandwidth and cache.",
     "Apple WWDC22 10159; github.com/philipturner/metal-benchmarks"),
    ("memory", "Prefer vector loads and stores (half4/float4, 8-16 contiguous bytes per thread): "
     "Apple's shader core 'prefers to load and store vector types directly'. Index with signed "
     "int: unsigned wraparound rules can stop the compiler vectorizing. MLX's elementwise "
     "kernels move 8 bytes per thread; its gemv uses 4-wide vectors.",
     "Apple WWDC20 'Optimize Metal Performance for Apple silicon Macs' (10632); Apple tech "
     "talk 'Metal Compute on MacBook Pro' (10580); MLX kernels utils.h, gemv.h"),
    ("memory", "Memory latency is hidden by other resident SIMD groups and by issuing several "
     "independent loads before using any of them (MLX's softmax loads 4 values per thread up "
     "front).",
     "Apple WWDC20 10632; Apple tech talk 111374; MLX kernels softmax.h, defines.h"),
    ("memory", "Every kernel launch carries a fixed cost: CPU-side encoding, a barrier between "
     "dependent dispatches, GPU ramp-up and tail. Small ops are therefore overhead-bound and "
     "fusing removes launches. There is no public per-launch figure for Metal; this tool "
     "measured about 3.2 us for an mx.fast.metal_kernel launch and 1.8 us for a compiled MLX "
     "launch on an M4 (chip.launch_us is this Mac's).",
     "MLX docs, 'Compilation'; Apple WWDC22 10159; this tool's M4 measurement"),
    ("memory", "Exchange and reduce inside a SIMD group with simd_sum / simd_shuffle (registers "
     "only, no barrier), then use one barrier to combine the SIMD groups' partials. Barriers "
     "cost time (this tool measured one after every threadgroup-memory load at 1.5x the "
     "load's time on an M4), but never remove a barrier that orders a threadgroup-memory "
     "write before another SIMD group reads it. simd_sum is not free either.",
     "Apple WWDC22 10159; github.com/philipturner/metal-benchmarks; this tool's M4 measurement"),
    ("memory", "Fusing a producer into its consumer saves writing and re-reading the "
     "intermediate. mx.compile already fuses chains of elementwise ops (its fusion stops at "
     "reductions and matmuls), so against a compiled baseline the room is in fusions across "
     "norms, reductions, softmax and matmul prologues or epilogues. Fusion can also lose: "
     "recomputing a producer inside every consumer tile, or register pressure that cuts "
     "occupancy.",
     "MLX compile.cpp (is_fusable), MLX docs 'Compilation'; llama.cpp discussion 17621; this "
     "tool's MetalBench run: 15 of 16 regions with nothing shipped were elementwise chains, "
     "and every win crossed a reduction or matmul"),

    # -- where the time goes ---------------------------------------------------
    ("performance", "A kernel takes at least the larger of bytes moved / achievable bandwidth and "
     "arithmetic / achievable peak, plus launch overhead. It can also sit below both limits "
     "because it has too little parallelism to hide latency; then more parallelism helps "
     "without cutting bytes or arithmetic. Work out which case holds (region.bound, "
     "roofline_ms, op_ms) before choosing a change.",
     "Williams, Waterman and Patterson, 'Roofline', CACM 2009; Horace He, 'Making Deep "
     "Learning Go Brrrr From First Principles'"),
    ("performance", "With few rows through a matmul (one token, a short prompt, a small batch), the "
     "weights are nearly all the bytes, so time is about weight bytes / bandwidth, and one "
     "read of each weight can serve several rows almost free. That holds only until "
     "arithmetic catches up: for low-bit weights on a fast chip, dequantizing can make it "
     "compute-bound at 3-4 rows. Check how the MLX kernel chosen for that row count reads "
     "the weights before assuming room.",
     "kipply, 'Transformer Inference Arithmetic'; Bonsai port notes (a 4-row 2-bit "
     "matrix-vector bound by FMAs, not weight reads)"),
    ("performance", "With many rows, each weight is reused and arithmetic dominates: speed comes "
     "from reuse inside registers and threadgroup memory (tiling) and from matrix "
     "instructions. Bigger tiles reuse each load more but hold more on-chip storage, so the "
     "best tile depends on the shape and the chip.",
     "standard GPU GEMM practice; Apple tech talk 111375; Apple M5 post (prefill is compute-bound)"),
    ("performance", "When too few outputs exist to fill the GPU, splitting the reduction across "
     "threadgroups (split-K, or two-pass attention) adds parallelism, at the cost of a second "
     "pass and a changed summation order, so it is not bit-exact. MLX already ships split-K "
     "quantized and GEMM kernels and two-pass vector attention: check whether MLX already "
     "splits that shape.",
     "Stanford CRFM, 'Flash-Decoding for long-context inference'; MLX kernels quantized.h "
     "(affine_qvm_split_k, affine_qmm_t_splitk), steel_gemm_splitk.h, sdpa_vector.h"),
    ("performance", "Rewriting one tuned MLX op on its own (matrix-vector, gather matmul, "
     "attention) rarely beats it where MLX has a well-matched path. Wins come from fusions "
     "across a matmul or reduction boundary, from shapes where MLX's tiles waste work, and "
     "from reads shared across rows.",
     "MTPLX docs, laguna-mlxfast-port/PORT_KERNEL_LEDGER.md"),
    ("performance", "Without mx.compile, some work already arrives fused: the mx.fast ops (rms_norm, "
     "rope, scaled_dot_product_attention), addmm, and quantized matmul (dequantization inside "
     "the kernel). A custom kernel can reuse MLX's own building blocks (steel GEMM or attention "
     "tiles, quantized helpers) by pasting the installed headers into its header without their "
     "#pragma once; that ties the kernel to the installed MLX version.",
     "MLX metal_kernel.cpp and custom_kernel.cpp (header prepended, no include path); MTPLX "
     "moe_sorted_gather_nax.py and oMLX minimax_m3_vl/msa.py inline steel headers"),
    ("performance", "Mixture of experts: when several tokens go to the same expert, sorting them by "
     "expert lets one loaded weight tile serve all that expert's rows (MLX's "
     "gather_qmm_rhs schedules tiles per expert). mlx-lm sorts only when there are at least "
     "64 routing indices, so one-token decode gets no sharing.",
     "mlx_lm models/switch_layers.py (do_sort = indices.size >= 64); MLX kernels quantized.h "
     "(affine_gather_qmm_rhs)"),
    ("performance", "Tunings and even correctness can change between chips: oMLX tunes size "
     "crossovers per chip (M1 Ultra vs M3 Ultra), reports a kernel that was exact elsewhere "
     "giving wrong rows on M3 Ultra, and works around MLX's own M5 gather kernel being wrong "
     "when K is not a multiple of 64. Judge memory- vs compute-bound from this Mac's measured "
     "chip numbers and the weights' bit width, not from another chip's results.",
     "oMLX source comments (deepseek_v4/switch_layers.py, row_exact_qmv.py, m5_gather_qmm.py); "
     "community evidence, not peer reviewed"),
    ("performance", "A speedup measured on one kernel alone usually shrinks in the whole model: its "
     "share of the step (Amdahl), different cache and overlap conditions, integration costs, "
     "and isolation baselines weaker than what the model really runs. Warm caches flatter "
     "isolated timings; only the harness's whole-model check counts.",
     "Rapid-MLX release notes v0.14.0; MTPLX laguna ledger (a 4.7x fusion measured against a "
     "naive baseline, 1.00-1.04x against the real one); Triton do_bench clears L2 for this reason"),

    # -- bit-exactness under MLX 0.32.4 ----------------------------------------
    ("exactness", "Float addition is not associative, so a bit-exact replacement must reproduce "
     "MLX's order exactly: which elements each thread owns, the per-thread loop, simd_sum at "
     "the same point over the same lanes, the same threadgroup combine. MLX picks threadgroup "
     "sizes from the input shape, so its order can change with size. The order inside "
     "simd_sum is unspecified: use the same intrinsic where MLX does, not a hand-written tree.",
     "Goldberg, 'What Every Computer Scientist Should Know About Floating-Point Arithmetic'; "
     "MLX kernels reduction/, rms_norm.metal"),
    ("exactness", "A fused multiply-add rounds once; a multiply then an add rounds twice. MLX's "
     "kernels and mx.fast.metal_kernel both compile in safe math mode, which still lets the "
     "compiler contract a*b+c inside one expression. Copy MLX's expressions as written; use "
     "fma() or '#pragma METAL fp contract(off)' when a specific behavior is needed.",
     "MLX kernels CMakeLists.txt (-fno-fast-math); MLX metal_kernel.h (MathMode::Safe "
     "default); Apple DTS, developer.apple.com/forums/thread/844598"),
    ("exactness", "Each MLX elementwise op rounds its result to the array's dtype, and mx.compile "
     "rounds every intermediate to its dtype too. To be bit-exact, convert to bfloat/half "
     "exactly where the original graph had an array; keeping float across those boundaries "
     "is more precise and therefore different.",
     "MLX kernels binary_ops.h, bf16_math.h; MLX compiled.cpp"),
    ("exactness", "Quantized matmul arithmetic depends on which kernel MLX picks, by M, K, N and "
     "GPU generation. qmv / qmv_fast / qmv_quad compute scale*sum(x*q) + bias*sum(x) in float; "
     "qmv_wide (two or more rows, on M3 and later) computes sum(x*(scale*q + bias)) in float; "
     "qmm rounds each dequantized weight to the compute type once, then accumulates in float. "
     "qmv_wide's order, which a copy must keep (affine, 2 to 5 rows: one tile of all the rows): "
     "8 lanes per output row; lane L takes groups L, L+8, L+16, ...; each group in 8-value "
     "chunks, first to last: decode the chunk as dequantize() does (float scale and bias, "
     "scale*(byte & mask) + bias with the scale pre-divided per position), then per row "
     "acc = 0, acc += x[i]*w[i] for i = 0..7, partial += acc; at the end partial += "
     "simd_shuffle_down by 4, then 2, then 1, and lane 0 writes. Summing across 32 lanes, "
     "simd_sum, or a different chunk length changes the result. Measured here (M4, 2-bit, "
     "K and N 4,096-12,288): a copy that keeps this order but gives each thread two output "
     "rows, sharing each loaded x chunk, was bit-identical and 1.7-2.2x faster than MLX's own "
     "at 2-3 rows. "
     "Steel GEMM and unquantized gemv accumulate in float and round once at the end.",
     "MLX kernels quantized.h (qdot, qmv_wide, QuantizedBlockLoader), steel/gemm/mma.h, "
     "steel/gemm/transforms.h, gemv.h; MLX quantized.cpp (dispatch_qmv, get_qmv_batch_limit)"),
    ("exactness", "rms_norm, rows up to 4,096: each thread sums the squares of 4 contiguous "
     "elements in float, then simd_sum, lane 0 to threadgroup memory, simd_sum again; "
     "inv = metal::precise::rsqrt(sum / axis_size + eps); out = w * T(x * inv), which rounds "
     "twice. Threadgroup size is 32*ceil(ceil(axis/4)/32); longer rows use a looped variant.",
     "MLX rms_norm.metal; MLX normalization.cpp"),
    ("exactness", "softmax, logsumexp and SDPA use fast::exp; mx.exp and sigmoid use precise::exp, "
     "so name the namespace explicitly. Default softmax on bfloat/half does its max, exp and "
     "sum in bfloat/half (only precise=True accumulates in float) and normalizes by "
     "multiplying by a reciprocal. bfloat/half mx.sum and mx.mean accumulate in bfloat/half.",
     "MLX kernels softmax.h, unary_ops.h, softmax.metal, reduce.metal; MLX ops.cpp"),
    ("exactness", "To learn an op's arithmetic, read its MLX kernel source and also which kernel and "
     "threadgroup size MLX selects for this shape and chip; the harness confirms "
     "bit-exactness on this device.",
     "MLX source at the installed version"),
    # -- what repeated winning work had in common (concepts, not recipes) -----
    ("diagnosis", "Find what limits a decode step before changing it. If the whole step runs close "
     "to its bytes divided by the chip's measured achievable bandwidth, only moving fewer bytes "
     "helps. Otherwise find the gap: kernels running below peak (limited by latency, or by "
     "instructions such as unpacking weights) or time between kernels (launches, chains of small "
     "dependent kernels, threadgroups that finish unevenly, the host). The limit moves as other "
     "parts get faster, so check again after each win.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("sharing", "Rows that read the same operand should share its load: batch rows, tokens routed "
     "to the same expert, query heads that share a key/value head, rows of a recurrence that share "
     "query/key loads, projections that read the same activation. One threadgroup loading a tile "
     "once and applying it to every row cuts traffic in proportion. A separate accumulator per "
     "row, summed in the one-row path's order, keeps each row bit-identical; computing rows with "
     "simdgroup matrix multiplies changes each row's summation order. More rows per thread raises "
     "register pressure, and past a point it spills or cuts occupancy and loses, so the rows per "
     "kernel have a limit.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("sharing", "With a few rows, a quantized matrix-vector kernel can be limited by unpacking "
     "weights rather than reading them: a kernel that unpacks the weights again for every row "
     "costs more with each row while its bytes stay flat. Unpack each weight once for all rows, "
     "and move terms that depend only on the activation (such as its sum, used with the offset) "
     "out of the weight loop, keeping each row's arithmetic unchanged. When a several-row product "
     "costs close to rows times the one-row cost, this is the opportunity.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); MLX quantized.h (qdot uses the activation sum with the bias)"),
    ("sharing", "Three things decide whether a several-row copy of MLX's one-row 4-bit qmv_fast costs "
     "about one row. (1) Keep qmv_fast's per-lane traversal exactly (which values each lane owns per "
     "block, products added in groups of four left to right, scale * accum + sum * bias joined block "
     "by block, simd_sum over the lanes) and change only what is computed once: each lane unpacks its "
     "weights of a block once and streams every row through them. (2) A 4-bit value n unpacks "
     "exactly and cheaply as the half (0x6400 | n) - 1024. qmv_fast pre-scales x by powers of 16 to "
     "skip shifts, but x * n is exact in float (an 8-bit mantissa times a 4-bit integer), so the "
     "unscaled product gives the same bits and needs no per-row pre-scaling. (3) Fully unroll the "
     "loops over rows and unpacked values so the weights stay in registers; left rolled, the "
     "compiler spills them and the kernel runs 2-3x slower. A hand-written kernel built this way "
     "measured, on an M4 with Qwen3.8-27B's projections, 1.04x one row's cost at 2 to 4 rows, 1.35x "
     "at 6 and 1.70x at 8 (oMLX's multi-row kernel: 1.17x, 1.75x, 2.42x and 3.1x), each row bit for "
     "bit the one-row call. Rounding also depends on the compiler, so check bits against the "
     "one-row call at every width.",
     "MLX 0.32.4 quantized.h (qmv_fast_impl, qdot, load_vector); this tool's M4 measurement of a "
     "hand-written kernel (2026-11-08)"),
    ("exactness", "A kernel that serves several rows must give each row exactly what the one-row "
     "path gives: the same operations, order and rounding points per element. Which thread "
     "computes an element, when, and through which memory, at the same precision, are free to "
     "change. The traps: a shape change that makes MLX choose a different kernel for some op "
     "(joining weight matrices or batching calls changes the sizes its kernel choice depends on), "
     "split partial sums whose partition or combine order differs from the reference's, tie rules "
     "in a selection that differ from the library's, and fused intermediates that skip a rounding "
     "the original made. A one-step difference written into a cache surfaces later as a different "
     "token, so check bits, not only the chosen token, at every width used.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); MLX quantized.cpp (get_qmv_batch_limit, qmm split-K), scaled_dot_product_attention.cpp"),
    ("layout", "Don't materialize a layout the consumer can read directly at the same speed. "
     "Copies, concatenations, transposes, gathers and slice-updates between two real operations "
     "cost bandwidth and launches without doing any math. Let the consumer index data where and "
     "in the order it is stored, let the producer write the final layout, and generate cheap "
     "values such as masks inside the kernel instead of reading them. Updating a slice of a "
     "buffer the graph still references can force a copy of the whole buffer. These changes are "
     "exact when they move or regenerate the same values without type changes and the consumer's "
     "kernel and arithmetic stay the same (MLX can pick a different kernel for a strided view than "
     "for a contiguous copy).",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); MLX kernel selection on row_contiguous inputs"),
    ("layout", "A producer should emit the small side data its consumer would otherwise compute "
     "again (a sum, a table, a transposed copy) while it holds the values in registers: one "
     "dispatch and one re-read fewer. It stays exact when the producer uses the order and types "
     "the consumer used.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("work", "Compute what many tiles share once, upstream, not again in every tile's prologue: "
     "repeated prologue work grows with the grid. Values that are the same every step, such as "
     "converted copies of scales or constant tables, belong outside the step (exact when the "
     "conversion matches the kernel's own). The reverse also holds: fusing a cheap prologue into "
     "a kernel with many tiles can make it slower.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("work", "Rows, outputs and tiles that nothing reads still cost full price: a prompt pass whose "
     "result is used only at its last row needs the vocabulary head only there, computed with the "
     "many-row path's arithmetic for that row (a one-row matmul can take a different kernel), and "
     "tiles whose output is masked or discarded can be skipped. Make the skip test the same for a "
     "whole SIMD group (tile finer rather than branching per thread), and keep every write a later "
     "step reads, such as cache and state writes. Producing only the chosen token instead of all "
     "logits changes the outputs a job checks, so it fits only where the check is on tokens.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("fusion", "Fusing pays when it removes a launch, a dependent stage or a memory round trip on "
     "the critical path, and independent calls of the same shape (per row, per head) can be "
     "batched into one grid if each call's arithmetic stays the same. It loses when the fused ops "
     "want different parallel shapes (a reduction wanting a few large threadgroups next to a "
     "matrix-vector wanting many), when it adds threadgroup memory or barriers, and once the GPU "
     "is busy end to end. Copy the library's formulas literally, including rounding each "
     "intermediate to the type the library stored it in.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("balance", "A launch ends when its slowest threadgroup ends. Look for one group doing extra "
     "serial work while others idle, partitions that are empty or ragged, and SIMD groups that "
     "could take an independent chain, and rebalance. Splitting a long floating-point "
     "accumulation changes its rounding, so split a walk only where the reference already splits "
     "it, with the same partition and combine order, or where the work is not a floating-point "
     "sum (integer work, max, independent outputs). The best grid and tile sizes move as the "
     "kernels around them change and differ between chips, so measure on the target rather than "
     "keep an old setting.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("latency", "Issue loads one or two iterations before they are used, keep the values in "
     "registers, use wide loads, and unroll short loops whose trip counts are known at compile "
     "time. A per-thread array indexed by a value the compiler cannot make constant (for example "
     "in a loop with a run-time trip count that cannot be fully unrolled) goes to slow stack "
     "memory; with compile-time trip counts it stays in registers. Several independent "
     "accumulators per thread, one per output, hide latency without reordering any output's sum. "
     "Register pressure is the limit: lookahead or extra rows that spill make it slower.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); Apple WWDC16 'Advanced Metal Shader Optimization' (606)"),
    ("latency", "Remove barriers that guard no live hazard. After simd_sum every active lane of that "
     "SIMD group holds the result, so using it within the SIMD group needs no barrier; a barrier "
     "is still needed when the result passes through threadgroup memory to other SIMD groups. "
     "Giving each stage its own scratch buffer removes only the reuse hazard, not the "
     "producer-to-consumer barrier, and a threadgroup of one SIMD group needs only "
     "simdgroup_barrier. Barriers tend to cost the most when only one or two threadgroups are "
     "resident per core.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); Apple WWDC16 606; MLX rms_norm.metal"),
    ("exactness", "Some rewrites are exact by arithmetic: multiplying by a power of two as long as no "
     "value overflows or becomes subnormal (when folding it into a per-group scale, fold it into "
     "the bias or activation-sum term too and keep the scale in range for its type); integer work, "
     "such as a counting sort that keeps the library's order for equal keys or a scatter that "
     "inverts a permutation; and max, when NaN and signed-zero handling match. Top-k results can be "
     "regrouped only with the library's tie rule and output order, since a later sum over them "
     "follows that order. Floating-point sums and products otherwise keep their order.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); IEEE 754; mlx_lm models (expert sums follow the top-k order)"),
    ("work", "Shape tiles to the data: in a sorted mixture-of-experts matmul, align tiles to each "
     "expert's run of rows; use narrow tiles when few columns are live; skip padded rows when that "
     "saves time, and never let them write output.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("latency", "A kernel can be limited by instruction issue or by a long dependent chain (typically "
     "an exact recurrence) rather than by memory or arithmetic throughput. Prefetching memory does "
     "not help there; shorten the chain: delete provably dead operations, move fixed per-step work "
     "off the chain or spread it over more outputs, and interleave independent chains per thread.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16); Apple WWDC16 606"),
)

# For the inference-loop search (identical: output): what the judge needs to rewrite a model's inference loop
# (its run()) so the outputs stay identical, grouped by mechanism. Checked the same way, 2026-10-16;
# regrouped 2026-10-20.
MODEL_PRINCIPLES = (
    ("compute once", "In diffusion-style loops, anything that depends only on the prompt (text "
     "encoder outputs) or only on the step (timestep embeddings, position or rotary tables) can be "
     "computed once per call instead of every step, exact if built with identical ops, dtype and "
     "shape. MLX's Stable Diffusion example already encodes the text once, before its step loop; "
     "check whether this model's loop does.",
     "diffusers pipelines (pipeline_stable_diffusion.py, pipeline_flux.py); mlx-examples "
     "stable_diffusion (_get_text_conditioning, generate_latents)"),
    ("batching", "Batching the conditional and unconditional halves of guidance into one pass "
     "changes the batch size and can change kernel choice, so it is at best within tolerance, and "
     "padding prompts of different lengths needs a mask. MLX's Stable Diffusion example already "
     "runs both halves in one UNet pass.",
     "diffusers pipelines (pipeline_stable_diffusion.py, pipeline_flux.py); mlx-examples "
     "stable_diffusion (_denoising_step)"),
    ("batching", "Batching several inputs raises throughput but can change per-row results, because "
     "kernel choice depends on the row count; padding is exact only with masks and pooling that "
     "exclude the padded positions. mlx-lm's batch generation left-pads prompts and hands each "
     "cache the padding so it masks those positions (the recurrent ArraysCache too; a "
     "RotatingKVCache with keep tokens is not supported). Batch only within a call. Each call must do "
     "its own work: nothing carries from one call to the next.",
     "MLX kernel dispatch (quantized.cpp); standard practice; mlx_lm generate.py (_make_cache) and "
     "models/cache.py (BatchKVCache, ArraysCache make_mask)"),
    ("overlap", "mlx-lm already dispatches step n+1 with mx.async_eval before reading step n's "
     "token, so the GPU never waits on Python; the computation is the same graph, so it stays "
     "bit-identical. Keep host synchronizations (.item(), .tolist(), mx.eval) to one decision per "
     "round.",
     "mlx_lm generate.py (generate_step)"),
    ("overlap", "Host time per step is critical-path time: building the graph, allocating and every "
     "blocking read delay the GPU. Keep host synchronizations to one per round, compute the "
     "round's decisions in one reduction on the device, keep draft tokens on the device, and "
     "submit finished parts of the graph early so the GPU starts while the host builds the rest. "
     "Outputs stay exact provided decisions made on the device compare the same values and moving "
     "evaluation points does not change what mx.compile fuses.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("skip discarded work", "mlx-lm's generate_step already skips logits nobody reads: it sends all "
     "but the last prompt token through the model in chunks and evaluates only the cache state, so "
     "those chunks' logits are never evaluated, then keeps the last position's logits only. A loop "
     "that evaluates outputs for rows it then discards (a model file's own run(), a pass sliced "
     "afterward) pays for them; computing the output head only on the rows that are read saves "
     "that, but fewer rows can select different kernels, so it is exact only if the outputs still "
     "match.",
     "mlx_lm generate.py (generate_step prefill loop and _step)"),
    ("exactness", "Reproduce the original's token choice exactly: mlx-lm takes the argmax "
     "of logits - logsumexp(logits) in the logits' dtype, and rounding after the subtraction can "
     "create ties that argmax breaks toward the lower index. Taking the argmax of raw logits, or "
     "changing their dtype, can change tokens.",
     "mlx_lm generate.py (generate_step and speculative_generate_step samplers)"),
    ("exactness", "Keep the original's prompt chunking (prefill_step_size): different chunk "
     "boundaries change reduction order and, for recurrent layers, the state. mx.compile promises "
     "results equal 'up to numerical precision', not bitwise: the harness decides whether a "
     "compiled step is exact.",
     "mlx_lm generate.py; Thinking Machines on chunked prefill; MLX docs 'Compilation'"),
    ("guess and check", "Greedy speculative decoding: a drafter proposes k tokens, the target "
     "scores all k+1 positions in one pass, the longest prefix where each draft token equals the "
     "target's own argmax is kept, and the target's next token is appended (at least one token per "
     "pass). In exact arithmetic the output equals plain greedy decoding. The speedup depends on "
     "how many draft tokens are accepted and on how much more a k+1-token pass costs than a "
     "1-token pass; on Apple GPUs that cost is not flat, so the textbook formula overstates it.",
     "Leviathan et al. 2023 (arXiv 2211.17192); Chen et al. 2023 (arXiv 2302.01318); mlx_lm "
     "generate.py speculative_generate_step"),
    ("guess and check", "A k+1-row verify pass on stock MLX runs different kernels from one-token "
     "decoding (quantized matmul: qmv for one row, qmv_wide for several on M3 and later, tiled qmm "
     "above a row limit; attention: other plans as the query count grows), so its logits differ in "
     "the last bits and near-tie tokens would flip. The harness removes that: every call into the "
     "model that reads several generated tokens computes each one with the one-token arithmetic "
     "(linear layers row by row in one multi-row kernel, attention one query at a time), so a "
     "correct verify pass gives exactly the one-token numbers and the cache entries plain decoding "
     "writes. That costs more than a stock pass: on an M4, Qwen3-8B and Qwen3.8-27B measured about "
     "1.05x one token for 2 tokens, 1.4x for 3 and 1.6-1.8x for 4 (stock: 1.0x, 1.05x, 1.3x); the "
     "briefing's exact_rows gives this model's own costs. Plain decoding's one-token calls are "
     "untouched.",
     "MLX quantized.cpp and scaled_dot_product_attention.cpp (v0.32.4); this tool's M4 measurements "
     "(runs/diagnostics/row_exact); oMLX row_exact_qmv"),
    ("guess and check", "Measured on an M4 with Qwen3.8-27B and its own multi-token head, exact: one "
     "guess per round (a 2-token verify) gave a median 1.38x over plain decoding (1.12-1.60x over "
     "six prompts, 0.78-0.97 guesses accepted), two guesses 1.25x; the third verify token costs more "
     "than it saves at these acceptance rates.",
     "this tool's M4 measurement (runs/diagnostics/decode_loop/mtp_exact)"),
    ("guess and check", "Never redo work the loop already has: the verify pass yields the next "
     "token, so the target need not run again after a rejection. When most rounds accept every "
     "draft, don't pay for the reject on every round: reuse what the verify pass already computed "
     "instead of running the model again, and for recurrent state keep the inputs and starting "
     "state and replay only the committed prefix after a reject, rather than saving a checkpoint at "
     "every position. Both are exact only if that computation matches the plain one-token path bit "
     "for bit (same per-step arithmetic and chunking). The chance that all drafts are accepted "
     "falls quickly with the draft length.",
     "mlx_lm generate.py; independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("guess and check", "Each extra draft position adds a draft step and a verify row. Extend "
     "only while the expected extra accepted tokens beat that cost, using acceptance per position "
     "estimated as you go and the verify cost measured per width; prefer widths that fill a "
     "kernel tile and stay below the library's kernel switch points, measured on the target chip "
     "(crossing one changes the arithmetic as well as the cost). Speculation pays only if tokens "
     "produced per round (accepted plus the bonus token) times one plain step's cost beat the "
     "round's cost: in a mixture of experts verified tokens touch different experts, so verify "
     "cost grows much faster with width than in a dense model, and on unpredictable text it often "
     "loses.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("drafters", "A smaller model of the same family works as a drafter only with the same tokenizer "
     "and vocabulary, since draft ids go straight into the target. mlx-lm's speculative step feeds "
     "the last accepted draft token back to the drafter, so no drafter pass is wasted. This tool "
     "measured about 1.4x with Qwen3-0.6B drafting for Qwen3-VL-8B (2 draft tokens) and 1.08x for "
     "Qwen3-4B on an M4: a drafter that is a large fraction of the target's cost gains little.",
     "mlx_lm generate.py; this tool's M4 measurements"),
    ("drafters", "Some checkpoints ship multi-token-prediction heads (DeepSeek-V3/R1, Qwen3-Next, "
     "Qwen3.5/3.6/3.8, GLM-4.5 family, MiMo, Nemotron-H); mlx-lm drops those weights at load, but a "
     "head split into its own checkpoint (an *_mtp model type) loads with mlx-vlm's "
     "speculative.drafters.load_drafter and attaches to the mlx-lm model with bind(model). A head "
     "drafts from the target's hidden state, the input to its output layer: call the module that "
     "holds the layers and then the output layer yourself (the same ops as the model's call), and "
     "build the head's memory of the conversation in read() from those hidden states. Prompt lookup "
     "(drafting from n-grams already in the context) needs no weights and wins only when the output "
     "copies the input (summaries, document QA, code edits); this tool measured it losing on "
     "ordinary chat.",
     "DeepSeek-V3 report (arXiv 2412.19437); mlx_lm models qwen3_5.py (MTP weights dropped); "
     "mlx_vlm speculative/drafters (qwen3_5_mtp: bind, prefill_from_target_hidden, draft_block, "
     "accept_verified_tokens); prompt-lookup-decoding README; this tool's measurement"),
    ("drafters", "Under exact-match verification (greedy decoding) the drafter affects only how many "
     "guesses are accepted, never the output, so it may be cheapened or improved freely: give it its own record of the committed context, "
     "score only the part of the vocabulary it can reach, or score coarsely and rerank a short "
     "list exactly. Weigh acceptance against its cost per step; a drafter that scores better "
     "offline can still lose.",
     "independent wins on several community optimization boards for MLX on Apple GPUs (Yukon mlxfast write-ups and code, read 2026-10-16)"),
    ("cache", "After a rejected draft the target's and drafter's caches must be rolled back. mlx-lm's "
     "KVCache trims in place; RotatingKVCache only while its offset is below its window; ArraysCache "
     "(the recurrent state of gated-delta and Mamba layers) cannot be trimmed, and mlx-lm then "
     "refuses speculation: keep a copy of its list before the verify (list(cache.state); the list "
     "itself is live and overwritten in place), put it back, trim the attention caches and re-read "
     "the accepted tokens with the model's own call, which the harness computes with one-token "
     "arithmetic, so the state matches plain decoding. trim_prompt_cache trims nothing, silently, if "
     "any cache cannot be trimmed.",
     "mlx_lm models/cache.py (trim, is_trimmable, trim_prompt_cache, ArraysCache.state); generate.py; "
     "this tool's M4 measurement (runs/diagnostics/decode_loop/mtp_exact)"),
    ("forbidden", "Approximate methods change outputs and are refused: step caches that reuse or "
     "extrapolate features across steps (DeepCache, TeaCache, PAB, FasterCache, FirstBlockCache, "
     "TaylorSeer, MagCache), fewer steps or another sampler, KV-cache quantization the original "
     "doesn't use, sparse or approximate prefill, accepting drafts without verifying them, "
     "Medusa's typical acceptance, and lower precision anywhere.",
     "diffusers docs 'Caching methods'; DeepCache (arXiv 2312.00858); TeaCache (arXiv 2411.19108); "
     "Medusa (arXiv 2401.10774)"),
)
