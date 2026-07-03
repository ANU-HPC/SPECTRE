// SPDX-FileCopyrightText: 2026 Proxima Fusion
//
// SPDX-License-Identifier: MIT
//
// PocketFFT (BSD-3-Clause) bridge for SPECTRE's 2D c2c FFT -- the only
// non-Fortran code in the FFT path, needed because PocketFFT is a
// header-only C++ library. Called from fft_backend_pocketfft_mod.F90
// through the matching bind(C) interface; compiled only for
// SPECTRE_FFT_BACKEND=pocketfft builds.
//
// Data layout matches the Fortran cplxin/cplxout(nt, nz) arrays:
// column-major, i.e. the nt ("theta") axis is contiguous/fastest and nz
// ("zeta") is slowest. Transforms are UNNORMALIZED with FFTW sign
// conventions -- dir > 0 forward (exp(-i...), FFTW_FORWARD), dir < 0
// backward (exp(+i...), FFTW_BACKWARD) -- so the normalization done in
// numerical_mod.F90 is unchanged. PocketFFT caches its 1D plans internally
// (thread-local get_plan cache), so repeated same-size calls reuse plans
// without an explicit planning step.

#include <complex>

#include "pocketfft_hdronly.h"

extern "C" void spectre_fft2d_c2c(int dir, int nt, int nz,
                                  const double *cin, double *cout) {
  using pocketfft::shape_t;
  using pocketfft::stride_t;
  const size_t elem = sizeof(std::complex<double>);

  // Fortran (nt, nz) column-major: nt contiguous (axis 1), nz outer (axis 0).
  shape_t shape{static_cast<size_t>(nz), static_cast<size_t>(nt)};
  stride_t stride{static_cast<ptrdiff_t>(nt * elem),
                  static_cast<ptrdiff_t>(elem)};
  shape_t axes{0, 1};

  const bool forward = (dir > 0);  // matches FFTW_FORWARD == exp(-i)
  auto in = reinterpret_cast<const std::complex<double> *>(cin);
  auto out = reinterpret_cast<std::complex<double> *>(cout);

  pocketfft::c2c(shape, stride, stride, axes, forward, in, out, 1.0);
}
