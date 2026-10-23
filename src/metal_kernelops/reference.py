import torch
"""Pure PyTorch selective scan for reference: explicit recurrence."""


def selective_scan(u, dt, A, B, C):
    """
    Reference implementation: batch x dim x length state-space recurrence.

    The loop is exact mamba_ssm's selective scan without the ``z`` gate or ``D``
    skip:

        x_t = exp(dt_t * A) * x_{t-1} + dt_t * B_t * u_t
        y_t = <x_t, C_t>

    Args:
        u, dt: (batch, dim, length) contiguous float32 tensors
        A: (dim, dstate) negative reals
        B, C: (batch, groups, dstate, length), groups dividing dim

    Returns:
        (batch, dim, length) output
    """
    batch, dim, length = u.shape
    groups, states = B.shape[1], A.shape[1]
    device = u.device
    dtype = u.dtype
    y = torch.zeros_like(u)
    x = torch.zeros((batch, dim, states), dtype=dtype, device=device)
    for t in range(length):
        u_t = u[..., t]  # (batch, dim)
        dt_t = dt[..., t]  # (batch, dim)
        B_t = B[:, :, :, t]  # (batch, groups, states)
        C_t = C[:, :, :, t]  # (batch, groups, states)
        if dim % groups != 0:
            raise ValueError("dim must be divisible by groups")
        per_group = dim // groups
        # expand A: (groups, per_group, states) -> (groups*per_group, states)
        A_expanded = A.repeat_interleave(groups, dim=0)  # (dim, states)
        for b in range(batch):
            for g in range(groups):
                for d in range(per_group):
                    channel = g * per_group + d
                    a_vec = A_expanded[channel]  # (states,)
                    a_scalar = torch.exp(a_vec * dt_t[b, channel])
                    # x[b, group*per_group+d] += dt_t[b,d]*u[b,d]*B[b,g,:,d]
                    term1 = a_scalar * x[b, channel]
                    term2 = dt_t[b, channel] * u_t[b, channel] * B_t[b, g]
                    x[b, channel] = term1 + term2
                    # C shape: (batch, groups, states)
                    y[b, channel, t] = torch.dot(x[b, channel].cpu().numpy(), C_t[b, g].cpu().numpy())
    return y.to(device).to(dtype)
