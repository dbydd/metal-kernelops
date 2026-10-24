"""Metal selective scan: fused forward, analytic backward.

Design
------
The sequence is split into chunks of ``CHUNK`` steps. Within a chunk every state
dimension advances independently, so one thread owns one ``(batch, channel,
state)`` lane and walks its 64 steps in registers. Across chunks the states
depend on each other, and that chain is what costs a second pass:

1. ``summarize`` reduces each chunk to the affine pair ``(incoming decay,
   incoming drive)`` that maps a chunk's entering state to its leaving state.
2. ``boundaries`` walks the chunk chain serially per lane, turning that pair
   list into the actual entering state of every chunk.
3. ``outputs`` replays each chunk from its now-known entering state.

Backward mirrors this: ``reverse_summaries`` reduces each chunk to the affine map
from a chunk's outgoing adjoint to its incoming adjoint, ``reverse_boundaries``
walks the chain backwards, and ``gradients`` replays each chunk from the end,
keeping the local states in registers.

The ``B`` and ``C`` gradients are the only cross-channel reductions. They meet in
threadgroup shared memory across ``BLOCK`` channels of one group, so a global
atomic fires once per threadgroup instead of once per channel.
"""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

CHUNK = 64
"""Steps advanced per chunk, and the size of the per-thread state history."""

BLOCK = 8
"""
Channels reduced together in one threadgroup by the backward pass. Must divide
the number of channels per group, so a threadgroup never straddles two groups.
"""


@lru_cache(maxsize=8)
def _library(block):
    return torch.mps.compile_shader(_SOURCE.replace("$BLOCK", str(block)))


def _pick_block(channels_per_group):
    """Largest divisor of the group width that stays within BLOCK."""
    width = min(BLOCK, channels_per_group)
    while channels_per_group % width:
        width -= 1
    return max(width, 1)


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint chunk_size = 64;

kernel void summarize(
    device const float* u, device const float* dt, device const float* A,
    device const float* B, device float* starts, device float* decays,
    constant uint& dim, constant uint& length, constant uint& groups,
    constant uint& states, constant uint& chunks,
    uint row [[threadgroup_position_in_grid]],
    uint n [[thread_index_in_threadgroup]]) {
    uint k = row % chunks, bd = row / chunks;
    uint d = bd % dim, b = bd / dim, g = d / (dim / groups);
    float av = n < states ? A[d * states + n] : 0.0f;
    float drive = 0.0f, log_decay = 0.0f;
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    if (n < states) {
        for (uint t = start; t < end; ++t) {
            uint p = bd * length + t;
            float delta = dt[p], a = exp(delta * av);
            float bv = B[((b * groups + g) * length + t) * states + n];
            drive = a * drive + delta * u[p] * bv;
            log_decay += delta * av;
        }
    }
    uint out = (bd * chunks + k) * states + n;
    if (n < states) {
        starts[out] = drive;
        decays[out] = exp(log_decay);
    }
}

kernel void boundaries(
    device float* starts, device const float* decays,
    constant uint& chunks, constant uint& states,
    uint row [[threadgroup_position_in_grid]],
    uint n [[thread_index_in_threadgroup]]) {
    if (n >= states) return;
    float running = 0.0f;
    for (uint k = 0; k < chunks; ++k) {
        uint p = (row * chunks + k) * states + n;
        float next = starts[p] + decays[p] * running;
        starts[p] = running;
        running = next;
    }
}

kernel void outputs(
    device const float* u, device const float* dt, device const float* A,
    device const float* B, device const float* C, device const float* starts,
    device float* y,
    constant uint& dim, constant uint& length, constant uint& groups,
    constant uint& states, constant uint& chunks,
    uint row [[threadgroup_position_in_grid]],
    uint n [[thread_index_in_threadgroup]]) {
    uint k = row % chunks, bd = row / chunks;
    uint d = bd % dim, b = bd / dim, g = d / (dim / groups);
    float av = n < states ? A[d * states + n] : 0.0f;
    float x = n < states ? starts[(bd * chunks + k) * states + n] : 0.0f;
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    for (uint t = start; t < end; ++t) {
        float yt = 0.0f;
        if (n < states) {
            uint p = bd * length + t;
            uint bc = ((b * groups + g) * length + t) * states + n;
            x = exp(dt[p] * av) * x + dt[p] * u[p] * B[bc];
            yt = x * C[bc];
        }
        float total = simd_sum(yt);
        if (n == 0) y[bd * length + t] = total;
    }
}

kernel void reverse_summaries(
    device const float* dt, device const float* A, device const float* C,
    device const float* dy, device float* adjoints,
    constant uint& dim, constant uint& length, constant uint& groups,
    constant uint& states, constant uint& chunks,
    uint row [[threadgroup_position_in_grid]],
    uint n [[thread_index_in_threadgroup]]) {
    uint k = row % chunks, bd = row / chunks;
    uint d = bd % dim, b = bd / dim, g = d / (dim / groups);
    float av = n < states ? A[d * states + n] : 0.0f, q = 0.0f;
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    if (n < states) {
        for (int t = int(end) - 1; t >= int(start); --t) {
            uint p = bd * length + t;
            uint bc = ((b * groups + g) * length + t) * states + n;
            q = (q + dy[p] * C[bc]) * exp(dt[p] * av);
        }
    }
    if (n < states) adjoints[(bd * chunks + k) * states + n] = q;
}

kernel void reverse_boundaries(
    device float* adjoints, device const float* decays,
    constant uint& chunks, constant uint& states,
    uint row [[threadgroup_position_in_grid]],
    uint n [[thread_index_in_threadgroup]]) {
    if (n >= states) return;
    float running = 0.0f;
    for (int k = int(chunks) - 1; k >= 0; --k) {
        uint p = (row * chunks + k) * states + n;
        float next = adjoints[p] + decays[p] * running;
        adjoints[p] = running;
        running = next;
    }
}

kernel void gradients(
    device const float* u, device const float* dt, device const float* A,
    device const float* B, device const float* C, device const float* starts,
    device const float* adjoints, device const float* dy,
    device float* du, device float* ddt, device float* dA,
    device atomic_float* dB, device atomic_float* dC,
    constant uint& dim, constant uint& length, constant uint& groups,
    constant uint& states, constant uint& chunks, constant uint& dgroups,
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float share_b[$BLOCK * 32];
    threadgroup float share_c[$BLOCK * 32];
    uint n = tid % 32, lane = tid / 32;
    uint k = row % chunks, rest = row / chunks;
    uint d = (rest % dgroups) * $BLOCK + lane, b = rest / dgroups;
    bool active = n < states && d < dim;
    uint g = d / (dim / groups);
    float av = active ? A[d * states + n] : 0.0f;
    uint bd = b * dim + d, boundary = (bd * chunks + k) * states + n;
    float entering = active ? starts[boundary] : 0.0f;
    float x = entering, history[64];
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    if (active) {
        for (uint t = start; t < end; ++t) {
            uint p = bd * length + t;
            float bv = B[((b * groups + g) * length + t) * states + n];
            x = exp(dt[p] * av) * x + dt[p] * u[p] * bv;
            history[t - start] = x;
        }
    }
    float q = active ? adjoints[boundary] : 0.0f, grad_a = 0.0f;
    for (int t = int(end) - 1; t >= int(start); --t) {
        uint p = bd * length + t;
        float grad_u = 0.0f, grad_dt = 0.0f, part_b = 0.0f, part_c = 0.0f;
        if (active) {
            uint bc = ((b * groups + g) * length + t) * states + n;
            float delta = dt[p], a = exp(delta * av);
            float previous = uint(t) == start ? entering : history[t - start - 1];
            float dh = q + dy[p] * C[bc];
            grad_u = dh * delta * B[bc];
            grad_dt = dh * (a * previous * av + u[p] * B[bc]);
            grad_a += dh * a * previous * delta;
            part_b = dh * delta * u[p];
            part_c = dy[p] * history[t - start];
            q = dh * a;
        }
        share_b[lane * 32 + n] = part_b;
        share_c[lane * 32 + n] = part_c;
        float total_u = simd_sum(grad_u), total_dt = simd_sum(grad_dt);
        if (n == 0 && d < dim) {
            du[p] = total_u;
            ddt[p] = total_dt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && n < states && d < dim) {
            float sum_b = 0.0f, sum_c = 0.0f;
            for (uint i = 0; i < $BLOCK; ++i) {
                sum_b += share_b[i * 32 + n];
                sum_c += share_c[i * 32 + n];
            }
            uint bc = ((b * groups + g) * length + t) * states + n;
            atomic_fetch_add_explicit(dB + bc, sum_b, memory_order_relaxed);
            atomic_fetch_add_explicit(dC + bc, sum_c, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) dA[boundary] = grad_a;
}
"""


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C):
        batch, dim, length = u.shape
        groups, states = B.shape[1], A.shape[1]
        chunks = (length + CHUNK - 1) // CHUNK
        u, dt, A = u.contiguous(), dt.contiguous(), A.contiguous()
        B, C = B.transpose(-1, -2).contiguous(), C.transpose(-1, -2).contiguous()
        starts = u.new_empty(batch, dim, chunks, states)
        decays = torch.empty_like(starts)
        y = torch.empty_like(u)
        lib = _library(min(BLOCK, dim // groups))
        shape = (dim, length, groups, states, chunks)
        lanes = {"threads": batch * dim * chunks * 32, "group_size": 32}
        chain = {"threads": batch * dim * 32, "group_size": 32}
        lib.summarize(u, dt, A, B, starts, decays, *shape, **lanes)
        lib.boundaries(starts, decays, chunks, states, **chain)
        lib.outputs(u, dt, A, B, C, starts, y, *shape, **lanes)
        ctx.save_for_backward(u, dt, A, B, C, starts, decays)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        u, dt, A, B, C, starts, decays = ctx.saved_tensors
        batch, dim, length = u.shape
        groups, states, chunks = B.shape[1], A.shape[1], starts.shape[2]
        dy = dy.contiguous()
        adjoints = torch.empty_like(starts)
        du, ddt = torch.empty_like(u), torch.empty_like(dt)
        dA = torch.empty_like(starts)
        dB, dC = torch.zeros_like(B), torch.zeros_like(C)
        block = _pick_block(dim // groups)
        dgroups = dim // block
        lib = _library(block)
        shape = (dim, length, groups, states, chunks)
        lanes = {"threads": batch * dim * chunks * 32, "group_size": 32}
        chain = {"threads": batch * dim * 32, "group_size": 32}
        lib.reverse_summaries(dt, A, C, dy, adjoints, *shape, **lanes)
        lib.reverse_boundaries(adjoints, decays, chunks, states, **chain)
        lib.gradients(u, dt, A, B, C, starts, adjoints, dy, du, ddt, dA, dB, dC,
                      *shape, dgroups,
                      threads=batch * dgroups * chunks * 32 * block,
                      group_size=32 * block)
        return du, ddt, dA.sum(dim=(0, 2)), dB.transpose(-1, -2), dC.transpose(-1, -2)


def scan_mps(u, dt, A, B, C):
    """Fused Metal selective scan on the MPS device. float32 only."""
    if u.device.type != "mps":
        raise ValueError("Metal kernels need tensors on the mps device")
    if u.dtype != torch.float32:
        raise ValueError(f"Metal kernels need float32, got {u.dtype}")
    for name, tensor in (("u", u), ("dt", dt), ("A", A), ("B", B), ("C", C)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return _Scan.apply(u, dt, A, B, C)


def selective_scan(u, dt, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False):
    """Drop-in selective scan matching mamba_ssm's ``selective_scan_fn``.

    Args:
        u, dt: (batch, dim, length)
        A: (dim, dstate), negative
        B, C: (batch, groups, dstate, length), groups dividing dim
        D: (dim,) skip connection
        z: (batch, dim, length) gate, applied as ``out * silu(z)``
        delta_bias: (dim,) added to dt
        delta_softplus: apply softplus to dt before the scan

    Returns:
        (batch, dim, length)
    """
    if delta_bias is not None:
        dt = dt + delta_bias[..., None]
    if delta_softplus:
        dt = torch.nn.functional.softplus(dt)
    out = scan_mps(u, dt, A, B, C)
    if D is not None:
        out = out + u * D[..., None]
    if z is not None:
        out = out * torch.nn.functional.silu(z)
    return out


selective_scan_fn = selective_scan


def scan_reference(u, dt, A, B, C):
    """Pure torch selective scan, evaluated step by step."""
    from .reference import selective_scan as reference_scan
    return reference_scan(u, dt, A, B, C)
