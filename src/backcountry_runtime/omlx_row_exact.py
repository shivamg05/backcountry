# SPDX-License-Identifier: Apache-2.0
# Copyright the oMLX authors (github.com/jundot/omlx, commit d762927).
# Modified by backcountry: the one-row-arithmetic multi-row quantized
# projection (row_exact_qmv.quantized_linear) and the qmv transcription it uses
# (moe_verify_gather), merged into one file; group, one-row, unrolled and MoE
# paths removed. Licensed under the Apache License, Version 2.0
# (LICENSE-omlx beside this file); unmodified copies of the sources are in
# runs/diagnostics/row_exact/omlx_vendor/ of the backcountry repo.
"""Multi-row quantized projection with one-row decode arithmetic per row.

Stock ``quantized_matmul`` switches kernels with the row count (``qmv_fast``
or ``qmv`` at one row, ``qmv_wide`` and then tiled qmm for more), so a row of
a several-token call and the one-token call for the same token get different
bits from the same projection. Here every row runs MLX 0.32.4's one-row
``qmv_fast`` or ``qmv`` traversal, so row ``r`` of the output equals
``quantized_matmul(x[r:r+1])`` bit for bit; each weight block is decoded once
and applied to all rows of a tile. Layouts the kernel does not cover run one
stock one-row call per row.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import mlx.nn as nn

_BITS = (4, 5, 6, 8)
_GROUP_SIZES = (32, 64, 128)

_HEADER = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;
constant constexpr int FAST = __FAST__;
constant constexpr int PACK_FACTOR =
    (BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS);
constant constexpr int BYTES_PER_PACK =
    ((BITS & (BITS - 1)) == 0) ? 4 : (BITS == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = FAST ? 2 : 1;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int BN = 8;

template <typename T>
inline float load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <typename T>
inline float load_vector_safe(const device T* x, thread float* x_thread, int N) {
  float sum = 0;
  if (BITS == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  for (int i = N; i < VALUES_PER_THREAD; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

inline float qdot_n(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  float accum = 0;
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (BITS == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (BITS == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (BITS == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}
"""


def qmv_fast_layout(k: int, n: int, bits: int, *, gathered: bool = False) -> bool:
    """MLX 0.32.4's one-row affine dispatch: K must align to the kernel block
    (512 for 4/5-bit, 256 for 6/8-bit). Only gathered QMV still requires N
    to be a multiple of 8; ordinary QMV supports partial output tiles."""
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return (not gathered or n % 8 == 0) and k % (
        pack_factor * (1 if bits == 2 else 2) * 32
    ) == 0

MAX_ROWS = 16
# Output columns per simdgroup (qmv's own tile: a threadgroup owns 8).
_RPS = 4
# Verify projections run in a dependent chain, so latency wins: below this
# many tiles every row gets its own threadgroups (rows of a tile are adjacent
# in the grid and hit cache); wider outputs (the vocabulary head) share each
# weight pass between rows, up to this many per-lane input values.
_SHARED_ROWS_MIN_TILES = 2048
_ROW_VALUES_BUDGET = 32
# Padding a partial row tile pays for large 8-bit weight streams. Smaller
# matrices and 4/5/6-bit unpackers are faster with the existing divisor tile.
_PADDED_ROWS_MIN_WEIGHT_BYTES = 128 * 1024 * 1024

# Per row this is qmv_fast_impl (FAST) or qmv_impl, exactly as
# moe_verify_gather runs them per (row, expert) pair; the row loop sits
# inside each K block so the block's weight bytes are fetched once. Columns
# past N (qmv's partial last tile) are skipped: each column's arithmetic does
# not depend on which tile computes it. ``ROW_EXACT_UNROLL`` precedes every
# row and column loop: empty, or a full unroll in the ``unrolled`` kernels
# (unrolling repeats each accumulator's steps in the same order).
_TILE = r"""
// Tiny scale/bias arrays (N < 8) arrive in the constant address space, so
// their pointer type is a template parameter.
template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename SP>
METAL_FUNC void row_exact_tile(
    const device T* x,
    const device uint32_t* w,
    SP scales,
    SP biases,
    device T* y,
    int tile,
    int row0,
    uint simd_gid,
    uint simd_lid) {
  constexpr bool PARTIAL = (N_OUT % (2 * RPS)) != 0;
  const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  const int in_vec_size_g = K_SIZE / GS;
  const int out_row = tile * (2 * RPS) + int(simd_gid) * RPS;
  if (PARTIAL && out_row >= N_OUT) {
    return;
  }
  const int valid = PARTIAL ? min(RPS, N_OUT - out_row) : RPS;

  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  SP sc = scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  SP bs = biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* xp = x + row0 * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

  float result[ROWS][RPS];
  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      result[i][r] = 0;
    }
  }

  int k = 0;
  const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
  for (; k < full_limit; k += BLOCK_SIZE) {
    float xs[ROWS][VALUES_PER_THREAD];
    float sums[ROWS];
    ROW_EXACT_UNROLL
    for (int i = 0; i < ROWS; i++) {
      sums[i] = load_vector<T>(xp + i * K_SIZE, xs[i]);
    }
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float wq[W_TERMS];
      decode_w(ws + r * in_vec_size_w, wq);
      float s = sc[r * in_vec_size_g];
      float b = bs[r * in_vec_size_g];
      ROW_EXACT_UNROLL
      for (int i = 0; i < ROWS; i++) {
        result[i][r] += s * wdot(wq, xs[i]) + sums[i] * b;
      }
    }
    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;
    xp += BLOCK_SIZE;
  }
  if (!FAST) {
    const int remaining = clamp(
        int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      ROW_EXACT_UNROLL
      for (int i = 0; i < ROWS; i++) {
        float x_thread[VALUES_PER_THREAD];
        float sum = load_vector_safe<T>(xp + i * K_SIZE, x_thread, remaining);
        ROW_EXACT_UNROLL
        for (int r = 0; r < RPS; r++) {
          if (PARTIAL && r >= valid) {
            break;
          }
          const device uint8_t* wl = ws + r * in_vec_size_w;
          float s = sc[r * in_vec_size_g];
          float b = bs[r * in_vec_size_g];
          result[i][r] += qdot_n(wl, x_thread, s, b, sum, remaining);
        }
      }
    }
  }

  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float v = simd_sum(result[i][r]);
      if (simd_lid == 0) {
        y[(row0 + i) * N_OUT + out_row + r] = static_cast<T>(v);
      }
    }
  }
}
"""

_SOURCE = r"""
    row_exact_tile<T, K_SIZE, N_SIZE, ROWS, RPS>(
        x, w, scales, biases, y,
        int(threadgroup_position_in_grid.z),
        int(threadgroup_position_in_grid.y) * ROWS,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


# qmv's ``qdot`` split in two: the weight terms of a full pack run are decoded
# once per K block and output column, then every row accumulates them against
# its own inputs. Each term and each accumulation step is qdot's, in qdot's
# order (the 5/6-bit runs cover at most two packs, so qdot's cumulative
# pointer steps are the 8i/5i and 4i/3i offsets used here).
_DECODED_DOT = r"""
constant constexpr int W_TERMS = (BITS == 5) ? 12 * VALUES_PER_THREAD / 8
    : ((BITS == 6) ? 6 * VALUES_PER_THREAD / 4 : VALUES_PER_THREAD);

inline void decode_w(const device uint8_t* w, thread float* wq) {
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      wq[4 * i] = ws[i] & 0x000f;
      wq[4 * i + 1] = ws[i] & 0x00f0;
      wq[4 * i + 2] = ws[i] & 0x0f00;
      wq[4 * i + 3] = ws[i] & 0xf000;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
      const device uint8_t* wb = w + 5 * i;
      thread float* wv = wq + 12 * i;
      wv[0] = wb[0] & 0x1f;
      wv[1] = wb[0] & 0xe0;
      wv[2] = wb[1] & 0x3;
      wv[3] = wb[1] & 0x7c;
      wv[4] = wb[1] & 0x80;
      wv[5] = wb[2] & 0xf;
      wv[6] = wb[2] & 0xf0;
      wv[7] = wb[3] & 0x1;
      wv[8] = wb[3] & 0x3e;
      wv[9] = wb[3] & 0xc0;
      wv[10] = wb[4] & 0x7;
      wv[11] = wb[4] & 0xf8;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const device uint8_t* wb = w + 3 * i;
      thread float* wv = wq + 6 * i;
      wv[0] = wb[0] & 0x3f;
      wv[1] = wb[0] & 0xc0;
      wv[2] = wb[1] & 0x0f;
      wv[3] = wb[1] & 0xf0;
      wv[4] = wb[2] & 0x03;
      wv[5] = wb[2] & 0xfc;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      wq[i] = w[i];
    }
  }
}

inline float wdot(const thread float* wq, const thread float* x) {
  float accum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      accum +=
          (x[4 * i] * wq[4 * i] + x[4 * i + 1] * wq[4 * i + 1] +
           x[4 * i + 2] * wq[4 * i + 2] + x[4 * i + 3] * wq[4 * i + 3]);
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
      const thread float* xv = x + 8 * i;
      const thread float* wv = wq + 12 * i;
      accum += wv[0] * xv[0];
      accum += wv[1] * xv[1];
      accum += wv[2] * (xv[1] * 256.0f);
      accum += wv[3] * xv[2];
      accum += wv[4] * xv[3];
      accum += wv[5] * (xv[3] * 256.0f);
      accum += wv[6] * xv[4];
      accum += wv[7] * (xv[4] * 256.0f);
      accum += wv[8] * xv[5];
      accum += wv[9] * xv[6];
      accum += wv[10] * (xv[6] * 256.0f);
      accum += wv[11] * xv[7];
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const thread float* xv = x + 4 * i;
      const thread float* wv = wq + 6 * i;
      accum += wv[0] * xv[0];
      accum += wv[1] * xv[1];
      accum += wv[2] * (xv[1] * 256.0f);
      accum += wv[3] * xv[2];
      accum += wv[4] * (xv[2] * 256.0f);
      accum += wv[5] * xv[3];
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      accum += x[i] * wq[i];
    }
  }
  return accum;
}
"""


def _header(bits: int, group_size: int, fast: bool, unrolled: bool = False) -> str:
    unroll = '_Pragma("clang loop unroll(full)")' if unrolled else ""
    return (
        f"#define ROW_EXACT_UNROLL {unroll}\n"
        + (_HEADER + _DECODED_DOT + _TILE)
        .replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )


@cache
def _kernel(bits: int, group_size: int, fast: bool, unrolled: bool = False):
    return mx.fast.metal_kernel(
        name=f"omlx_row_exact_qmv_b{bits}_gs{group_size}_{int(fast)}" + ("_u" if unrolled else ""),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_header(bits, group_size, fast, unrolled),
        source=_SOURCE,
    )


def _values_per_thread(bits: int, fast: bool) -> int:
    return (8 if bits == 5 else (4 if bits == 6 else 32 // bits)) * (2 if fast else 1)


def _kernel_supported(linear: nn.QuantizedLinear, x: mx.array) -> bool:
    bits, group_size = linear.bits, linear.group_size
    if (
        getattr(linear, "mode", "affine") != "affine"
        or bits not in _BITS
        or group_size not in _GROUP_SIZES
        or linear.biases is None
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return False
    n = int(linear.weight.shape[0])
    k = int(linear.scales.shape[-1]) * group_size
    fast = qmv_fast_layout(k, n, bits)
    return (
        x.shape[-1] == k
        and k % 8 == 0
        and group_size % _values_per_thread(bits, fast) == 0
    )


def _launch_geometry(
    bits: int, fast: bool, n: int, rows: int, k: int
) -> tuple[int, int]:
    """(rows per threadgroup, output columns per simdgroup) for one launch."""
    if (n + 2 * _RPS - 1) // (2 * _RPS) < _SHARED_ROWS_MIN_TILES:
        return 1, _RPS
    limit = _ROW_VALUES_BUDGET // _values_per_thread(bits, fast)
    rows_per_group = next(
        (d for d in range(min(rows, limit), 1, -1) if rows % d == 0), 1
    )
    if (
        rows_per_group == 1
        and bits == 8
        and fast
        and n * k >= _PADDED_ROWS_MIN_WEIGHT_BYTES
    ):
        rows_per_group = min(rows, limit)
    return rows_per_group, _RPS


class _Plan:
    """Static launch parameters of one (linear, row count, dtype) shape."""

    __slots__ = (
        "kernel",
        "template",
        "grid",
        "output_shapes",
        "output_dtypes",
        "bias",
        "padding",
    )

    def __init__(self, linear: nn.QuantizedLinear, x: mx.array, rows: int):
        k = int(x.shape[-1])
        n = int(linear.weight.shape[0])
        bits = int(linear.bits)
        fast = qmv_fast_layout(k, n, bits)
        rows_per_group, rps = _launch_geometry(bits, fast, n, rows, k)
        self.padding = -rows % rows_per_group
        padded_rows = rows + self.padding
        self.kernel = (bits, int(linear.group_size), fast)  # looked up per call, see _plan
        self.template = [
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", rows_per_group),
            ("RPS", rps),
        ]
        self.grid = (
            32,
            2 * (padded_rows // rows_per_group),
            (n + 2 * rps - 1) // (2 * rps),
        )
        self.output_shapes = [(padded_rows, n)]
        self.output_dtypes = [x.dtype]
        self.bias = "bias" in linear


_PLANS: dict = {}


def _plan(linear: nn.QuantizedLinear, x: mx.array, rows: int) -> _Plan | None:
    # backcountry: plans are kept here by layout, not on the module (a
    # module's Python state is part of what graph-delivered kernels check),
    # and name their kernel rather than hold it, so the kernel object is
    # created again after a cache clear (under the tracer, which records
    # kernels created while it is installed). Weights are read per call.
    key = (rows, x.dtype, x.shape[-1], int(linear.weight.shape[0]), linear.bits, linear.group_size,
           getattr(linear, "mode", "affine"), linear.biases is None, linear.scales.dtype, "bias" in linear)
    if key not in _PLANS:
        supported = rows <= MAX_ROWS and _kernel_supported(linear, x)
        _PLANS[key] = _Plan(linear, x, rows) if supported else None
    return _PLANS[key]


def quantized_linear(linear: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    Layouts the kernel does not cover (non-affine modes, other bit widths or
    group sizes, more than MAX_ROWS rows) run one stock one-row call per row,
    which is the serial decode call itself.
    """
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    if rows <= 1:
        return linear(x)
    flat = x.reshape(rows, x.shape[-1])
    plan = _plan(linear, x, rows)
    if plan is None:
        y = mx.concatenate(
            [linear(flat[r : r + 1][None])[0] for r in range(rows)], axis=0
        )
        return y.reshape(*lead, -1)
    if plan.padding:
        flat = mx.pad(flat, [(0, plan.padding), (0, 0)])
    y = _kernel(*plan.kernel)(
        inputs=[flat, linear.weight, linear.scales, linear.biases],
        template=plan.template,
        grid=plan.grid,
        threadgroup=(32, 2, 1),
        output_shapes=plan.output_shapes,
        output_dtypes=plan.output_dtypes,
    )[0]
    if plan.padding:
        y = y[:rows]
    if plan.bias:
        y = y + linear["bias"]
    return y.reshape(*lead, -1)

