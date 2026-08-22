FROM ubuntu:24.04

LABEL org.opencontainers.image.title="Unofficial ROCm Windows Training Toolkit - WSL2" \
      org.opencontainers.image.description="Source-built HIP extension validation through WSL2 /dev/dxg" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/shumaimai/ROCm-Windows-Training-Toolkit"

ARG GPU_ARCH=gfx1101
ARG ROCM_VERSION=7.14.0
ARG TORCH_VERSION=2.12.0+rocm7.14.0
ARG ROCDXG_VERSION=1.2.2

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/workspace \
    HSA_ENABLE_DXG_DETECTION=1 \
    GPU_ARCH=${GPU_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git libatomic1 libnuma1 libquadmath0 \
      ninja-build python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
      --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
      "torch[device-${GPU_ARCH}]==${TORCH_VERSION}" \
      "rocm[devel,device-${GPU_ARCH}]==${ROCM_VERSION}" \
      numpy && \
    rocm-sdk init

# ROCDXG is the AMD userspace bridge for the WSL /dev/dxg device. Extract only
# its runtime and device-ID configuration; no host or model material is copied.
RUN curl -fsSL -o /tmp/rocdxg.deb \
      "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" && \
    dpkg-deb -x /tmp/rocdxg.deb /tmp/rocdxg && \
    install -D -m 0755 \
      "/tmp/rocdxg/opt/rocm/lib/librocdxg.so.${ROCDXG_VERSION}" \
      "/usr/lib/librocdxg.so.${ROCDXG_VERSION}" && \
    ln -s "librocdxg.so.${ROCDXG_VERSION}" /usr/lib/librocdxg.so.1 && \
    ln -s "librocdxg.so.1" /usr/lib/librocdxg.so && \
    install -D -m 0644 /tmp/rocdxg/opt/rocm/share/rocdxg/dids.conf \
      /usr/share/rocdxg/dids.conf && \
    ln -s /usr/lib/x86_64-linux-gnu/libnuma.so.1 \
      /usr/lib/x86_64-linux-gnu/libnuma.so && \
    ldconfig && rm -rf /tmp/rocdxg /tmp/rocdxg.deb

WORKDIR /workspace
COPY . /workspace
RUN chmod +x /workspace/docker/wsl2/entrypoint.sh

ENTRYPOINT ["/workspace/docker/wsl2/entrypoint.sh"]
CMD ["validate"]
