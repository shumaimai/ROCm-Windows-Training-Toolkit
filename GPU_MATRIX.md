# Radeon validation matrix

This matrix tracks source build and synthetic kernel validation. It does not
replace AMD's released ROCm compatibility matrix.

## Validated

| Architecture | Representative GPU | Status |
|---|---|---|
| gfx1101 | Radeon RX 7800 XT | Windows build and synthetic tests pass |

## Requested release-validation targets

Use an AMD-released Windows ROCm configuration supported for the exact GPU and
report all versions.

| Architecture | Representative GPU | Issue |
|---|---|---|
| gfx1100 | Radeon RX 7900 XTX | #4 |
| gfx1102 | Radeon RX 7600 | #7 |
| gfx1150 | Ryzen AI 9 HX 375 / Radeon 890M | #6 |
| gfx1151 | Ryzen AI Max+ PRO 395 / Radeon 8060S | #8 |
| gfx1200 | Radeon RX 9060 XT | #5 |
| gfx1201 | Radeon AI PRO R9700 or Radeon RX 9070 XT | #9 |

## Requested TheRock nightly/community targets

These are not claims of released Windows product support. Use the exact
TheRock package date and commit, and treat failures as development evidence.

| Architecture | Representative GPU | Issue |
|---|---|---|
| gfx1030 | Radeon RX 6800 XT | #12 |
| gfx1031 | Radeon RX 6700 XT | #11 |
| gfx1032 | Radeon PRO W6600 or Radeon RX 6600 XT | #10 |

## Required report fields

- GPU model and VRAM
- `gcnArchName`
- Windows build and AMD driver
- Python, PyTorch, HIP/ROCm versions
- released ROCm or TheRock nightly channel/date
- commit SHA and WaveTrain build identity
- build output and first failing synthetic test
- confirmation that no private data, credentials, model files, checkpoints,
  adapters, or generated binaries are attached

Official references:

- https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
- https://github.com/ROCm/TheRock/blob/main/RELEASES.md
