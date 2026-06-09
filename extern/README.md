# FFT backend shim and notes

SPECTRE selects one FFT backend at build time via the `SPECTRE_FFT_BACKEND`
CMake option (see `../CMakeLists.txt`). This directory holds only SPECTRE's own
thin shim; the third-party FFT libraries are either fetched at configure time or
found on the system — none are vendored in-tree.

| Directory | Source | License | Used for |
|-----------|--------|---------|----------|
| `fft_shim/` | SPECTRE (BergTop Fusion) | MIT | uniform `spectre_fft2d_c2c` 2D c2c entry point (non-FFTW backends) |

## Backends

| Backend | Source | How obtained | License | Notes |
|---------|--------|--------------|---------|-------|
| `pocketfft` (default) | [pocketfft (cpp)](https://github.com/mreineck/pocketfft), M. Reinecke et al. | CMake `FetchContent`, pinned commit + SHA256 — downloads the single header `pocketfft_hdronly.h` | BSD-3-Clause | double precision, arbitrary sizes, machine-precise; fully MIT-redistributable |
| `fftw` | system FFTW | found on system | GPL-2.0+ | opt-in reference; **not** MIT-redistributable |
| `mkl` | Intel oneMKL FFTW3 interface | found on system (`fftw3.f03` + `libmkl_rt`) | Intel Simplified Software License | opt-in; redistributable binaries, commercial-use OK; reuses the FFTW3-ABI Fortran path |

**PocketFFT** is the recommended permissive default — it is not vendored; the
build fetches the single pinned header at configure time. For offline or
air-gapped builds, point CMake at a local copy instead of downloading:

```
pip install -e . -C cmake.define.FETCHCONTENT_SOURCE_DIR_POCKETFFT=/path/to/dir
```

where `/path/to/dir` contains `pocketfft_hdronly.h`. The pocketfft license is
retained in that header; SPECTRE's shim is MIT (see file headers).
