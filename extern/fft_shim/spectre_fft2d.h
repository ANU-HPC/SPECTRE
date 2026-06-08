/* SPDX-FileCopyrightText: 2026 BergTop Fusion
 *
 * SPDX-License-Identifier: MIT
 *
 * Uniform 2D complex-to-complex FFT entry point used by SPECTRE's Fortran core
 * (fft_backend_mod.F90). One implementation is compiled per build, selected by
 * the SPECTRE_FFT_BACKEND CMake option:
 *   - spectre_fft2d_pocketfft.cpp  (PocketFFT, BSD-3-Clause)
 *   - spectre_fft2d_pffft.c        (PFFFT double, FFTPACK/BSD-3-Clause)
 *
 * Data layout matches the Fortran cplxin/cplxout(nt, nz) arrays: column-major,
 * i.e. the nt ("theta") axis is contiguous/fastest and nz ("zeta") is slowest.
 * Transforms are UNNORMALIZED with FFTW sign conventions:
 *   dir > 0  -> forward  (exp(-i...), same as FFTW_FORWARD)
 *   dir < 0  -> backward (exp(+i...), same as FFTW_BACKWARD)
 * so the surrounding numerical_mod.F90 normalization is unchanged.
 */

#ifndef SPECTRE_FFT2D_H
#define SPECTRE_FFT2D_H

#ifdef __cplusplus
extern "C" {
#endif

/* In-/out-of-place 2D c2c transform of an (nt x nz) complex(double) grid. */
void spectre_fft2d_c2c(int dir, int nt, int nz, const double *cin, double *cout);

/* Returns 1 if the backend supports the given grid size, 0 otherwise.
 * PocketFFT supports any size; PFFFT requires nt and nz to be 2/3/5-smooth
 * multiples of its SIMD size. */
int spectre_fft2d_supported(int nt, int nz);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRE_FFT2D_H */
