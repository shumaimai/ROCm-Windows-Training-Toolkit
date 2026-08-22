# Linux Docker

This path targets native Linux hosts with the AMDGPU/KFD driver and exposes
`/dev/kfd` plus `/dev/dri` to a Linux container.

```bash
GPU_ARCH=gfx1101 docker compose -f docker/compose.linux.yml \
  run --rm --build validate
```

Unlike WSL2, this path does not use `/dev/dxg`, DXCore, or ROCDXG. Adjust
`GPU_ARCH` to the exact target reported by `rocminfo` and use a ROCm release
supported for the host GPU and distribution.
