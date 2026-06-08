// SPDX-FileCopyrightText: 2026 BergTop Fusion
//
// SPDX-License-Identifier: MIT
//
// PocketFFT backend for SPECTRE's 2D c2c transform. PocketFFT (BSD-3-Clause)
// caches its 1D plans internally (thread-local get_plan cache), so repeated
// same-size calls reuse plans -- a fair comparison against FFTW's persistent
// plan. See spectre_fft2d.h for the data-layout / sign conventions.

#include <complex>
#include "pocketfft_hdronly.h"
#include "spectre_fft2d.h"

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

extern "C" int spectre_fft2d_supported(int nt, int nz) {
  (void)nt;
  (void)nz;
  return 1;  // PocketFFT handles arbitrary sizes.
}
