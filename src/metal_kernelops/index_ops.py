"""Missing indexing paths expressed through native MPS gather and scatter."""

import torch

_CPP_NAME = {
    torch.bool: "bool", torch.uint8: "unsigned char", torch.int8: "signed char",
    torch.int16: "short", torch.int32: "int", torch.int64: "long long",
    torch.float16: "c10::Half", torch.bfloat16: "c10::BFloat16", torch.float32: "float",
    torch.float64: "double", torch.complex64: "c10::complex<float>",
    torch.complex128: "c10::complex<double>",
}


def _dtype_name(dtype):
    return _CPP_NAME.get(dtype, str(dtype).replace("torch.", ""))


def take(self, index):
    if index.dtype != torch.int64:
        raise RuntimeError("take(): Expected a long tensor for index")
    return self.reshape(-1)[index].reshape(index.shape)


def take_out(self, index, *, out):
    if out.dtype != self.dtype:
        raise RuntimeError(
            f"take(): self and out expected to have the same dtype, but got self.dtype = "
            f"{_dtype_name(self.dtype)} and out.dtype = {_dtype_name(out.dtype)}"
        )
    if torch._debug_has_internal_overlap(out) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor refers "
            "to a single memory location. Please clone() the tensor before performing the "
            "operation."
        )
    if torch._C._overlaps(self, out):
        raise RuntimeError(
            "unsupported operation: some elements of the input tensor and the written-to "
            "tensor refer to a single memory location. Please clone() the tensor before "
            "performing the operation."
        )
    value = take(self, index)
    out.resize_(value.shape)
    return out.copy_(value)


def channel_shuffle(self, groups):
    if self.ndim < 3:
        raise RuntimeError("channel_shuffle expects input with > 2 dims")
    if groups <= 0 or self.shape[1] % groups:
        raise RuntimeError("groups must be positive and divide the channel count")
    shape = self.shape
    out = (
        self.reshape(shape[0], groups, shape[1] // groups, *shape[2:])
        .transpose(1, 2)
        .reshape(shape)
        .contiguous()
    )
    return out.clone() if groups in (1, shape[1]) else out


def put_(self, index, source, accumulate=False):
    if index.dtype != torch.int64:
        raise RuntimeError("put_(): Expected a long tensor for index")
    if source.dtype != self.dtype:
        raise RuntimeError("put_(): self and source expected to have the same dtype")
    if source.numel() != index.numel():
        raise RuntimeError("put_(): Expected source and index to have the same number of elements")
    flat = self.reshape(-1)
    flat.index_put_((index.reshape(-1),), source.reshape(-1), accumulate=accumulate)
    if not self.is_contiguous():
        self.copy_(flat.reshape(self.shape))
    return self


def put(self, index, source, accumulate=False):
    return put_(self.clone(memory_format=torch.preserve_format), index, source, accumulate)


def put_out(self, index, source, accumulate=False, *, out):
    if out.dtype != self.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_dtype_name(self.dtype)}, but got "
            f"{_dtype_name(out.dtype)} instead"
        )
    value = put(self, index, source, accumulate)
    out.resize_(value.shape)
    return out.copy_(value)


def narrow_copy_out(self, dim, start, length, *, out):
    if out.dtype != self.dtype:
        raise RuntimeError("Expected self.dtype() == output.dtype() to be true, but got false.")
    value = self.narrow(dim, start, length)
    out.resize_(value.shape)
    return out.copy_(value)


OVERLAYS = (
    ("take", take), ("take.out", take_out),
    ("channel_shuffle", channel_shuffle),
    ("put_", put_), ("put", put), ("put.out", put_out),
    ("narrow_copy.out", narrow_copy_out),
)
