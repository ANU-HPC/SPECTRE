import pytest
import numpy as np

from spectre import spec_fft_fast


@pytest.mark.parametrize("nt_nz", [(196, 148), (183, 148), (183, 147)])
def test_fft_fast(nt_nz):

    nt, nz = nt_nz
    mpol = None
    ntor = None
    tarr = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zarr = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    tg, zg = np.meshgrid(tarr, zarr)

    arr = (
        np.cos(tg - zg)
        - 3.0 * np.sin(2 * tg + zg)
        + 1.4 * np.cos(3 * tg + 2 * zg)
        + 0.01 * np.sin(3 * tg + 1 * zg)
    )
    arr2 = np.cos(3 * tg + 2 * zg) - 4 * np.sin(3 * tg + 1 * zg)

    cust_even, cust_odd, im_, in_ = spec_fft_fast(
        tarr, zarr, np.array([arr, arr2]), mpol, ntor
    )

    mask = (im_ == 1) & (in_ == 1)
    assert np.allclose(cust_even[0][mask], 1.0)
    mask2 = (im_ == 3) & (in_ == -2)
    assert np.allclose(cust_even[0][mask2], 1.4)
    assert np.allclose(cust_even[0][~(mask | mask2)], 0.0)

    mask = (im_ == 2) & (in_ == -1)
    assert np.allclose(cust_odd[0][mask], -3.0)
    mask2 = (im_ == 3) & (in_ == -1)
    assert np.allclose(cust_odd[0][mask2], 0.01)
    assert np.allclose(cust_odd[0][~(mask | mask2)], 0.0)

    mask = (im_ == 3) & (in_ == -1)
    assert np.allclose(cust_odd[1][mask], -4.0)
    assert np.allclose(cust_odd[1][~mask], 0.0)

    mask = (im_ == 3) & (in_ == -2)
    assert np.allclose(cust_even[1][mask], 1.0)
    assert np.allclose(cust_even[1][~mask], 0.0)


@pytest.mark.parametrize("nt_nz", [(124, 1), (1, 146)])
def test_fft_fast_1d(nt_nz):

    nt, nz = nt_nz
    mpol = None
    ntor = None
    tarr = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zarr = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    tg, zg = np.meshgrid(tarr, zarr)

    arr = (
        np.cos(tg - zg)
        - 3.0 * np.sin(2 * tg + 2*zg)
        + 1.4 * np.cos(3 * tg + 2 * zg)
        + 0.01 * np.sin(3 * tg + 3 * zg)
    )
    arr2 = np.cos(3 * tg + 2 * zg) - 4 * np.sin(3 * tg + 1 * zg)

    cust_even, cust_odd, im_, in_ = spec_fft_fast(
        tarr, zarr, np.array([arr, arr2]), mpol, ntor
    )

    if nz == 1:
        mask = (im_ == 1)
        assert np.allclose(cust_even[0][mask], 1.0)
        mask2 = (im_ == 3)
        assert np.allclose(cust_even[0][mask2], 1.4)
        assert np.allclose(cust_even[0][~(mask | mask2)], 0.0)

        mask = (im_ == 2)
        assert np.allclose(cust_odd[0][mask], -3.0)
        mask2 = (im_ == 3)
        assert np.allclose(cust_odd[0][mask2], 0.01)
        assert np.allclose(cust_odd[0][~(mask | mask2)], 0.0)

        mask = (im_ == 3)
        assert np.allclose(cust_odd[1][mask], -4.0)
        assert np.allclose(cust_odd[1][~mask], 0.0)

        mask = (im_ == 3)
        assert np.allclose(cust_even[1][mask], 1.0)
        assert np.allclose(cust_even[1][~mask], 0.0)
    else:
        mask = (in_ == 1)
        assert np.allclose(cust_even[0][mask], 1.0)
        mask2 = (in_ == 2)
        assert np.allclose(cust_even[0][mask2], 1.4)
        assert np.allclose(cust_even[0][~(mask | mask2)], 0.0)

        mask = (in_ == 2)
        assert np.allclose(cust_odd[0][mask], 3.0)
        mask2 = (in_ == 3)
        assert np.allclose(cust_odd[0][mask2], -0.01)
        assert np.allclose(cust_odd[0][~(mask | mask2)], 0.0)

        mask = (in_ == 1)
        assert np.allclose(cust_odd[1][mask], 4.0)
        assert np.allclose(cust_odd[1][~mask], 0.0)

        mask = (in_ == 2)
        assert np.allclose(cust_even[1][mask], 1.0)
        assert np.allclose(cust_even[1][~mask], 0.0)