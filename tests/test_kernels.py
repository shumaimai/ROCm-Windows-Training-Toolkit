from __future__ import annotations

import torch
import torch.nn.functional as functional

from rocm_windows_training.hip_extension import make_autograd_function, make_conv_autograd_function


def reference_ssd(x, dt, a, b, c, d, z, bias, seq_idx):
    state = torch.zeros(
        x.shape[0], x.shape[2], x.shape[3], b.shape[-1],
        device=x.device, dtype=torch.float32,
    )
    outputs = []
    for token in range(x.shape[1]):
        if token and seq_idx.numel():
            state = torch.where(
                (seq_idx[:, token - 1] != seq_idx[:, token])[:, None, None, None],
                torch.zeros_like(state), state,
            )
        delta = functional.softplus(dt[:, token].float() + bias.float())
        decay = torch.exp(delta * a.float())
        state = (
            state * decay[..., None, None]
            + delta[..., None, None]
            * x[:, token].float()[..., None]
            * b[:, token].float()[..., None, :]
        )
        value = torch.einsum("bhdn,bhn->bhd", state, c[:, token].float())
        value += x[:, token].float() * d.float()[None, :, None]
        outputs.append((value * functional.silu(z[:, token].float()))[:, None])
    return torch.cat(outputs, dim=1).to(x.dtype)


def reference_conv(x, weight, seq_idx):
    outputs = torch.zeros_like(x)
    width = weight.shape[-1]
    for target in range(x.shape[-1]):
        raw = torch.zeros_like(x[:, :, target])
        for kernel_index in range(width):
            source = target - (width - 1 - kernel_index)
            if source < 0:
                continue
            valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            for boundary in range(source + 1, target + 1):
                valid &= seq_idx[:, boundary] == seq_idx[:, boundary - 1]
            raw += torch.where(valid[:, None], x[:, :, source], 0) * weight[:, 0, kernel_index]
        outputs[:, :, target] = functional.silu(raw)
    return outputs


def check_fp32() -> None:
    torch.manual_seed(101)
    seq = torch.tensor([[0, 0, 1, 1, 1]], device="cuda", dtype=torch.int32)
    tensors = [
        torch.randn(1, 5, 2, 3, device="cuda", requires_grad=True),
        torch.randn(1, 5, 2, device="cuda", requires_grad=True),
        (-torch.rand(2, device="cuda")).requires_grad_(),
        torch.randn(1, 5, 2, 4, device="cuda", requires_grad=True),
        torch.randn(1, 5, 2, 4, device="cuda", requires_grad=True),
        torch.randn(2, device="cuda", requires_grad=True),
        torch.randn(1, 5, 2, 3, device="cuda", requires_grad=True),
        torch.randn(2, device="cuda", requires_grad=True),
    ]
    ssd = make_autograd_function()
    actual = ssd(*tensors, seq)
    expected = reference_ssd(*tensors, seq)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward(retain_graph=True)
    actual_grads = [value.grad.detach().clone() for value in tensors]
    for value in tensors:
        value.grad = None
    expected.sum().backward()
    for left, right in zip(actual_grads, tensors):
        torch.testing.assert_close(left, right.grad, rtol=1e-5, atol=1e-6)

    x = torch.randn(1, 7, 5, device="cuda", requires_grad=True)
    weight = torch.randn(7, 1, 4, device="cuda", requires_grad=True)
    conv_seq = torch.tensor([[0, 0, 1, 1, 1]], device="cuda", dtype=torch.int32)
    conv = make_conv_autograd_function()
    actual_conv = conv(x, weight, conv_seq)
    expected_conv = reference_conv(x, weight, conv_seq)
    torch.testing.assert_close(actual_conv, expected_conv, rtol=1e-5, atol=1e-6)
    actual_conv.sum().backward(retain_graph=True)
    gradients = (x.grad.detach().clone(), weight.grad.detach().clone())
    x.grad = None
    weight.grad = None
    expected_conv.sum().backward()
    torch.testing.assert_close(gradients[0], x.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradients[1], weight.grad, rtol=1e-5, atol=1e-6)


def closed_form_low_precision(dtype: torch.dtype) -> None:
    torch.manual_seed(103)
    x = torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    dt = torch.randn(1, 2, 1, device="cuda", dtype=dtype, requires_grad=True)
    a = (-torch.rand(1, device="cuda")).requires_grad_()
    b = torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    d = torch.randn(1, device="cuda", requires_grad=True)
    z = torch.randn(1, 2, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(1, device="cuda", requires_grad=True)
    values = (x, dt, a, b, c, d, z, bias)
    empty_seq = torch.empty(0, device="cuda", dtype=torch.int32)
    actual = make_autograd_function()(*values, empty_seq)
    expected = reference_ssd(*values, empty_seq)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=8e-3, atol=8e-3)


def main() -> None:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("A native ROCm PyTorch GPU is required")
    check_fp32()
    closed_form_low_precision(torch.float16)
    closed_form_low_precision(torch.bfloat16)
    torch.cuda.synchronize()
    print("ROCm Windows Training HIP kernel tests passed")


if __name__ == "__main__":
    main()
