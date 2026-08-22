FROM ubuntu:24.04

LABEL org.opencontainers.image.title="Unofficial ROCm Windows Training Toolkit - Linux" \
      org.opencontainers.image.description="Source-built HIP extension validation on native Linux ROCm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/shumaimai/ROCm-Windows-Training-Toolkit"

ARG GPU_ARCH=gfx1101
ARG ROCM_VERSION=7.14.0
ARG TORCH_VERSION=2.12.0+rocm7.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/workspace \
    GPU_ARCH=${GPU_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git libatomic1 libnuma1 libquadmath0 ninja-build \
      python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
      --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
      "torch[device-${GPU_ARCH}]==${TORCH_VERSION}" \
      "rocm[devel,device-${GPU_ARCH}]==${ROCM_VERSION}" \
      numpy && \
    rocm-sdk init && \
    ln -s /usr/lib/x86_64-linux-gnu/libnuma.so.1 \
      /usr/lib/x86_64-linux-gnu/libnuma.so

WORKDIR /workspace
COPY . /workspace
RUN chmod +x /workspace/docker/linux/entrypoint.sh

ENTRYPOINT ["/workspace/docker/linux/entrypoint.sh"]
CMD ["validate"]
