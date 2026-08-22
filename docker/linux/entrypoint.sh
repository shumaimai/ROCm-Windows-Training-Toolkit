#!/usr/bin/env bash
set -euo pipefail

check_runtime() {
  test -c /dev/kfd || { echo "missing native Linux ROCm device: /dev/kfd" >&2; return 1; }
  test -d /dev/dri || { echo "missing native Linux DRM devices: /dev/dri" >&2; return 1; }
  python - <<'PY'
import json
import torch

report = {"torch": torch.__version__, "hip": torch.version.hip, "available": torch.cuda.is_available()}
if report["available"]:
    props = torch.cuda.get_device_properties(0)
    report.update(gpu=props.name, arch=props.gcnArchName, vram=props.total_memory)
print(json.dumps(report, indent=2))
if not report["available"]:
    raise SystemExit("PyTorch cannot access the native Linux ROCm device")
PY
}

case "${1:-validate}" in
  validate)
    check_runtime
    python /workspace/tests/test_kernels.py
    ;;
  shell)
    check_runtime
    exec bash
    ;;
  check)
    check_runtime
    ;;
  *) exec "$@" ;;
esac
