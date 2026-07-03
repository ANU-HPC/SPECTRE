# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

from spectre import _build_info
from spectre.field_tracer import RK_BACKENDS, default_rk_backend


def test_build_info_records_fft_backend():
    assert _build_info.fft_backend in ("fftw", "mkl", "pocketfft")
    assert isinstance(_build_info.commercial_use, bool)


def test_default_rk_backend_respects_build():
    backend = default_rk_backend()
    assert backend in RK_BACKENDS
    if _build_info.commercial_use:
        assert backend == "scipy"
