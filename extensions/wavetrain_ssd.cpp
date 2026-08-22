// Copyright 2026 Shuhei
// Licensed under the Apache License, Version 2.0.

#include <torch/extension.h>

std::vector<torch::Tensor> wavetrain_ssd_forward_hip(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor z,
    torch::Tensor dt_bias,
    torch::Tensor seq_idx);

std::vector<torch::Tensor> wavetrain_ssd_forward(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor z,
    torch::Tensor dt_bias,
    torch::Tensor seq_idx) {
  TORCH_CHECK(x.is_cuda(), "x must be on a HIP device");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(dt.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "SSD inputs must be contiguous");
  TORCH_CHECK(x.size(3) > 0 && b.size(3) > 0 && b.size(3) <= 64,
              "SSD state dimension must be in [1, 64]");
  return wavetrain_ssd_forward_hip(x, dt, a, b, c, d, z, dt_bias, seq_idx);
}

std::vector<torch::Tensor> wavetrain_ssd_backward_hip(
    torch::Tensor grad_output,
    torch::Tensor states,
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor z,
    torch::Tensor dt_bias,
    torch::Tensor seq_idx);

std::vector<torch::Tensor> wavetrain_ssd_backward(
    torch::Tensor grad_output,
    torch::Tensor states,
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor z,
    torch::Tensor dt_bias,
    torch::Tensor seq_idx) {
  return wavetrain_ssd_backward_hip(
      grad_output.contiguous(), states, x, dt, a, b, c, d, z, dt_bias, seq_idx);
}

torch::Tensor wavetrain_conv_forward_hip(
    torch::Tensor x, torch::Tensor weight, torch::Tensor seq_idx);
std::vector<torch::Tensor> wavetrain_conv_backward_hip(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weight,
    torch::Tensor seq_idx, bool need_weight_grad);

torch::Tensor wavetrain_conv_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor seq_idx) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "conv input must be contiguous HIP tensor");
  TORCH_CHECK(weight.size(2) <= 8, "causal convolution width must be <= 8");
  return wavetrain_conv_forward_hip(x, weight.contiguous(), seq_idx.contiguous());
}

std::vector<torch::Tensor> wavetrain_conv_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weight,
    torch::Tensor seq_idx, bool need_weight_grad) {
  return wavetrain_conv_backward_hip(
      grad_output.contiguous(), x, weight.contiguous(), seq_idx.contiguous(),
      need_weight_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &wavetrain_ssd_forward, "SSD-style fused HIP forward");
  module.def("backward", &wavetrain_ssd_backward, "SSD-style fused HIP backward");
  module.def("conv_forward", &wavetrain_conv_forward, "Causal conv HIP forward");
  module.def("conv_backward", &wavetrain_conv_backward, "Causal conv HIP backward");
}
