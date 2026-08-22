# Third-party software

WaveTrain Windows is source-only and does not vendor or redistribute the
following dependencies. Users install them separately under their own
licenses.

- AMD ROCm / HIP: MIT
  - https://github.com/ROCm/ROCm
  - https://github.com/ROCm/HIP
- PyTorch: BSD-style license and additional notices
  - https://github.com/pytorch/pytorch/blob/main/LICENSE
  - https://github.com/pytorch/pytorch/blob/main/NOTICE
- bitsandbytes: MIT
  - https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE
- Transformers: Apache-2.0
  - https://github.com/huggingface/transformers/blob/main/LICENSE
- PEFT: Apache-2.0
  - https://github.com/huggingface/peft/blob/main/LICENSE
- Mamba: Apache-2.0
  - https://github.com/state-spaces/mamba/blob/main/LICENSE
- causal-conv1d: BSD-3-Clause
  - https://github.com/Dao-AILab/causal-conv1d/blob/main/LICENSE

PLaMo model materials are not included. PLaMo 2 1B is separately distributed
under Apache-2.0:

- https://huggingface.co/pfnet/plamo-2-1b

The recurrence implementation cites the Mamba-2 / SSD publication:

- https://arxiv.org/abs/2405.21060

Re-evaluate notices before publishing prebuilt wheels, containers, runtime
DLLs, copied headers, model files, or bundled dependencies.
