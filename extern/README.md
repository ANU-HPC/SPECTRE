# FFT backend bridge

SPECTRE's Fortran core performs its 2D FFTs through the `fft_backend_mod`
module, which has one Fortran implementation file per backend family under
`fortran_src/` — exactly one is compiled, selected at build time by the
`SPECTRE_FFT_BACKEND` CMake option (see the top-level `CMakeLists.txt` and
`fortran_src/CMakeLists.txt`):

| Backend | Implementation | Library obtained | License | Notes |
|---------|----------------|------------------|---------|-------|
| `pocketfft` (default) | `fft_backend_pocketfft_mod.F90` + `fft_shim/spectre_fft2d_pocketfft.cpp` | [pocketfft (cpp)](https://github.com/mreineck/pocketfft), M. Reinecke et al.; single header fetched at configure time via CMake `FetchContent`, pinned commit + SHA256 | BSD-3-Clause | arbitrary sizes, machine-precise |
| `fftw` | `fft_backend_fftw_mod.F90` (pure Fortran, `fftw3.f03`) | system FFTW | GPL-2.0+ | fastest; **not** MIT-redistributable |
| `mkl` | `fft_backend_fftw_mod.F90` (same source) | system Intel oneMKL (`fftw3.f03` + `libmkl_rt`) | Intel Simplified Software License | proprietary but redistributable |

This directory holds only the pocketfft bridge (`fft_shim/`): PocketFFT is a
header-only C++ library with no Fortran bindings, so its backend needs one
small C++ file exposing the transform through a `bind(C)` interface. It is
the only non-Fortran code in the FFT path and is not compiled at all for the
`fftw`/`mkl` backends.

No third-party FFT code is vendored in-tree; libraries are either found on the
system (`fftw`, `mkl`) or fetched pinned at configure time (`pocketfft`). For
offline or air-gapped `pocketfft` builds, point CMake at a local copy of the
header instead of downloading:

```
pip install -e . -C cmake.define.FETCHCONTENT_SOURCE_DIR_POCKETFFT=/path/to/dir
```

where `/path/to/dir` contains `pocketfft_hdronly.h`. The pocketfft license is
retained in that header; SPECTRE's bridge itself is MIT.
