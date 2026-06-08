/* SPDX-FileCopyrightText: 2026 BergTop Fusion
 *
 * SPDX-License-Identifier: MIT
 *
 * PFFFT (double) backend for SPECTRE's 2D c2c transform. PFFFT is a 1D engine,
 * so the separable 2D transform is done as row transforms (length nt) followed
 * by column transforms (length nz). PFFFT keeps no internal plan cache, so we
 * cache the per-axis setups and aligned scratch buffers for the current grid
 * size -- fft_init only changes size occasionally, so this stays warm across
 * the many same-size calls SPECTRE makes (fair vs FFTW's persistent plan).
 *
 * PFFFT requires SIMD-aligned buffers and 2/3/5-smooth sizes that are multiples
 * of its SIMD width; spectre_fft2d_supported() reports whether a size is usable.
 * See spectre_fft2d.h for layout / sign conventions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pffft_double.h"
#include "spectre_fft2d.h"

/* One-entry cache for the current (nt, nz) grid. */
static int cache_nt = 0;
static int cache_nz = 0;
static PFFFTD_Setup *setup_t = NULL;  /* length nt */
static PFFFTD_Setup *setup_z = NULL;  /* length nz */
static double *buf_in = NULL;         /* aligned, 2*max(nt,nz) */
static double *buf_out = NULL;        /* aligned, 2*max(nt,nz) */
static double *work = NULL;           /* aligned, 2*max(nt,nz) */

static void ensure_setup(int nt, int nz) {
  if (nt == cache_nt && nz == cache_nz) return;

  if (!pffftd_is_valid_size(nt, PFFFT_COMPLEX) ||
      !pffftd_is_valid_size(nz, PFFFT_COMPLEX)) {
    fprintf(stderr,
            "SPECTRE pffft backend: grid (nt=%d, nz=%d) is not a PFFFT-supported "
            "size (each axis must be a 2/3/5-smooth multiple of the SIMD width). "
            "Rebuild with -DSPECTRE_FFT_BACKEND=pocketfft for arbitrary sizes.\n",
            nt, nz);
    abort();
  }

  if (setup_t) pffftd_destroy_setup(setup_t);
  if (setup_z) pffftd_destroy_setup(setup_z);
  if (buf_in) pffftd_aligned_free(buf_in);
  if (buf_out) pffftd_aligned_free(buf_out);
  if (work) pffftd_aligned_free(work);

  setup_t = pffftd_new_setup(nt, PFFFT_COMPLEX);
  setup_z = pffftd_new_setup(nz, PFFFT_COMPLEX);

  int nmax = (nt > nz) ? nt : nz;
  buf_in = (double *)pffftd_aligned_malloc(sizeof(double) * 2 * (size_t)nmax);
  buf_out = (double *)pffftd_aligned_malloc(sizeof(double) * 2 * (size_t)nmax);
  work = (double *)pffftd_aligned_malloc(sizeof(double) * 2 * (size_t)nmax);

  cache_nt = nt;
  cache_nz = nz;
}

int spectre_fft2d_supported(int nt, int nz) {
  return pffftd_is_valid_size(nt, PFFFT_COMPLEX) &&
         pffftd_is_valid_size(nz, PFFFT_COMPLEX);
}

void spectre_fft2d_c2c(int dir, int nt, int nz, const double *cin,
                       double *cout) {
  ensure_setup(nt, nz);
  pffft_direction_t pd = (dir > 0) ? PFFFT_FORWARD : PFFFT_BACKWARD;

  /* Step 1: transform each row (length nt). Rows are contiguous blocks of
   * 2*nt doubles. Output written into cout. */
  for (int iz = 0; iz < nz; ++iz) {
    const double *row = cin + (size_t)iz * 2 * nt;
    memcpy(buf_in, row, sizeof(double) * 2 * (size_t)nt);
    pffftd_transform_ordered(setup_t, buf_in, buf_out, work, pd);
    memcpy(cout + (size_t)iz * 2 * nt, buf_out, sizeof(double) * 2 * (size_t)nt);
  }

  /* Step 2: transform each column (length nz), gathering the strided column
   * out of cout, transforming, and scattering back. */
  for (int it = 0; it < nt; ++it) {
    for (int iz = 0; iz < nz; ++iz) {
      size_t idx = ((size_t)iz * nt + it) * 2;
      buf_in[2 * iz] = cout[idx];
      buf_in[2 * iz + 1] = cout[idx + 1];
    }
    pffftd_transform_ordered(setup_z, buf_in, buf_out, work, pd);
    for (int iz = 0; iz < nz; ++iz) {
      size_t idx = ((size_t)iz * nt + it) * 2;
      cout[idx] = buf_out[2 * iz];
      cout[idx + 1] = buf_out[2 * iz + 1];
    }
  }
}
