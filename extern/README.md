# Vendored third-party FFT libraries

These directories contain unmodified upstream sources for SPECTRE's permissive
FFT backends, plus a thin SPECTRE shim. One backend is selected at build time
via the `SPECTRE_FFT_BACKEND` CMake option (see `../CMakeLists.txt`). All code
here is under permissive, MIT-compatible licenses so SPECTRE can be redistributed
under MIT.

| Directory | Upstream | License | Used for |
|-----------|----------|---------|----------|
| `pocketfft/` | [pocketfft (cpp)](https://github.com/mreineck/pocketfft), M. Reinecke et al. | BSD-3-Clause | `SPECTRE_FFT_BACKEND=pocketfft` (default) |
| `pffft/` | [pffft (marton78 fork)](https://github.com/marton78/pffft), J. Pommier / NCAR FFTPACK | BSD-3-Clause / FFTPACK | `SPECTRE_FFT_BACKEND=pffft` |
| `fft_shim/` | SPECTRE (BergTop Fusion) | MIT | uniform `spectre_fft2d_c2c` 2D c2c entry point |

Two further backends are **not** vendored here (external dependencies, selected by the same
`SPECTRE_FFT_BACKEND` option):

| Backend | Source | License | Notes |
|---------|--------|---------|-------|
| `fftw` | system FFTW | GPL-2.0+ | opt-in reference; **not** MIT-redistributable |
| `mkl` | Intel oneMKL FFTW3 interface | Intel Simplified Software License (proprietary) | redistributable binaries, commercial-use OK; reuses the FFTW3-ABI Fortran path via MKL's `fftw3.f03` + `libmkl_rt`. Opt-in; not bundled in-source, not MIT |

The license text for each library is retained in its source headers
(`pocketfft/pocketfft_hdronly.h`) or as `pffft/LICENSE.txt`.

Notes:
- **PocketFFT** is double precision, supports arbitrary transform sizes, and is
  the recommended permissive default.
- **PFFFT** (double) only supports 2/3/5-smooth multiples of its SIMD width and
  loses precision (~1e-8) on radix-3/5 sizes; it is provided as an opt-in and is
  not suitable as a general default. See `../benchmark/report` for the analysis.
- **FFTW** (`SPECTRE_FFT_BACKEND=fftw`) is *not* vendored: it is found on the
  system and is GPL, so it is an opt-in reference only, not for MIT redistribution.
