from typing import Any
import torch
import triton
import triton.language as tl


def rmsnorm_grad_weight(x, weight, grad_out, eps):
    """
    x: ... x H
    weight: H
    grad_out: ... x H
    """
    x = x.view(-1, x.shape[-1])
    grad_out = grad_out.view(-1, grad_out.shape[-1])
    norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    grad_weight = torch.sum(x * grad_out / norm, dim=0)
    return grad_weight


def rmsnorm_grad_x(x, weight, grad_out, eps):
    """
    x: ... x H
    weight: H
    grad_out: ... x H
    """
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    grad_out = grad_out.view(-1, grad_out.shape[-1])
    norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    grad_x = grad_out * weight / norm - x * torch.sum(grad_out * x, dim=-1, keepdim=True) / (norm ** 3)
    return grad_x.view(*x_shape)

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        """
        x: ... x H
        weight: H
        """
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return x * weight.view(1, -1) / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def backward(ctx, grad_out):
        pass


@triton.jit
def rmsnorm_forward(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    output_ptr: tl.pointer_type,
    H: tl.uint32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    norm_factor = tl.sqrt(tl.sum(row * row) / H + eps)
    output = row * weight / norm_factor
    output_ptrs = output_ptr + row_idx * x_row_stride + offsets
    tl.store(output_ptrs, output, mask=mask)


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        """
        x: ... x H
        weight: H
        """
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        H = x.shape[-1]
        x_shape = x.shape

        x = x.view(-1, H)

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty_like(x, device=x.device)

        n_rows = x.shape[0]
        rmsnorm_forward[(n_rows,)](
            x,
            weight,
            x.stride(0),
            y,
            H,
            eps,
            num_warps=16,
            BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return y.view(*x_shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError("RMSNorm backward not implemented")