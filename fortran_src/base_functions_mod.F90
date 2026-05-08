! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module base_functions_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: get_cheby, get_cheby_d2, get_zernike, get_zernike_d2, get_zernike_rm

contains

  subroutine get_cheby(lss, lrad, cheby)
    real(rkind), intent(in) :: lss
    integer, intent(in) :: lrad
    real(rkind), intent(inout) :: cheby(0:lrad, 0:1)

    integer :: ll

    cheby = 0.0_rkind

    cheby(0, 0:1) = [1.0_rkind, 0.0_rkind]
    cheby(1, 0:1) = [lss, 1.0_rkind]
    do ll = 2, lrad
      cheby(ll, 0:1) = [2.0_rkind * lss * cheby(ll - 1, 0) - cheby(ll - 2, 0), 2.0_rkind * cheby(ll - 1, 0) + 2.0_rkind &
          &* lss * cheby(ll - 1, 1) - cheby(ll - 2, 1)]
    end do

    ! basis recombination
    do ll = 1, lrad
      cheby(ll, 0) = cheby(ll, 0) - (-1)  ** ll
    end do

    do ll = 0, lrad
      cheby(ll, 0:1) = cheby(ll, 0:1) / real(ll + 1, rkind)! scale for better conditioning
    end do

    return
  end subroutine get_cheby

  subroutine get_cheby_d2(lss, lrad, cheby)
    real(rkind), intent(in) :: lss
    integer, intent(in) :: lrad
    real(rkind), intent(inout) :: cheby(0:lrad, 0:2)

    integer :: ll

    cheby = 0.0_rkind

    cheby(0, 0:2) = [1.0_rkind, 0.0_rkind, 0.0_rkind] ! T_0: Chebyshev initialization: function, 1st-derivative,
    ! 2nd-derivative
    cheby(1, 0:2) = [lss, 1.0_rkind, 0.0_rkind] ! T_1: Chebyshev initialization: function, 1st-derivative,
    ! 2nd-derivative
    do ll = 2, lrad
      cheby(ll, 0:2) = [2.0_rkind * lss * cheby(ll - 1, 0) - cheby(ll - 2, 0), 2.0_rkind * cheby(ll - 1, 0) + 2.0_rkind &
          &* lss * cheby(ll - 1, 1) - cheby(ll - 2, 1), 2.0_rkind * cheby(ll - 1, 1) + 2.0_rkind * cheby(ll - 1, 1) +&
          & 2.0_rkind * lss * cheby(ll - 1, 2) - cheby(ll - 2, 2)]
    end do

    do ll = 1, lrad
      cheby(ll, 0) = cheby(ll, 0) - (-1)  ** ll
    end do

    do ll = 0, lrad
      cheby(ll, 0:2) = cheby(ll, 0:2) / real(ll + 1, rkind)! scale for better conditioning
    end do

    return
  end subroutine get_cheby_d2

  subroutine get_zernike(r, lrad, mpol, zernike)
    real(rkind), intent(in) :: r
    integer, intent(in) :: lrad, mpol
    real(rkind), intent(inout) :: zernike(0:lrad, 0:mpol, 0:1)

    real(rkind) :: rm, rm1 ! r to the power of m'th and m-1'th
    real(rkind) :: factor1, factor2, factor3, factor4
    integer :: m, n ! Zernike R^m_n

    rm = 1.0_rkind ! r to the power of m'th
    rm1 = 0.0_rkind ! r to the power of m-1'th
    zernike(:, :, :) = 0.0_rkind
    do m = 0, mpol
      if (lrad >= m) then
        zernike(m, m, 0:1) = [rm, real(m, rkind) * rm1]
      end if

      if (lrad >= m + 2) then
        zernike(m + 2, m, 0) = real(m + 2, rkind) * rm * r ** 2 - real(m + 1, rkind) * rm
        zernike(m + 2, m, 1) = real( (m + 2)  ** 2, rkind) * rm * r - real( (m + 1) * m, rkind) * rm1
      end if

      do n = m + 4, lrad, 2
        factor1 = real(n, rkind) / real(n ** 2 - m ** 2, rkind)
        factor2 = real(4 * (n - 1) , rkind)
        factor3 = real( (n - 2 + m)  ** 2, rkind) / real(n - 2, rkind) + real( (n - m)  ** 2, rkind) / real(n, rkind)
        factor4 = real( (n - 2)  ** 2 - m ** 2, rkind) / real(n - 2, rkind)

        zernike(n, m, 0) = factor1 * ( (factor2 * r ** 2 - factor3) * zernike(n - 2, m, 0) - factor4 * zernike(n - 4, m, 0))
        zernike(n, m, 1) = factor1 * (2.0_rkind * factor2 * r * zernike(n - 2, m, 0) + (factor2 * r ** 2 - factor3) *&
            & zernike(n - 2, m, 1) - factor4 * zernike(n - 4, m, 1))
      end do

      rm1 = rm
      rm = rm * r

    end do

    do n = 2, lrad, 2
      zernike(n, 0, 0) = zernike(n, 0, 0) - (-1)  **  (n / 2)
    end do

    if (mpol >= 1) then
      do n = 3, lrad, 2
        zernike(n, 1, 0) = zernike(n, 1, 0) - (-1)  **  ( (n - 1) / 2) * real( (n + 1) / 2, rkind) * r
        zernike(n, 1, 1) = zernike(n, 1, 1) - (-1)  **  ( (n - 1) / 2) * real( (n + 1) / 2, rkind)
      end do
    end if

    do m = 0, mpol
      do n = m, lrad, 2
        zernike(n, m, :) = zernike(n, m, :) / real(n + 1, rkind)
      end do
    end do
  end subroutine get_zernike

  subroutine get_zernike_d2(r, lrad, mpol, zernike)
    real(rkind), intent(in) :: r
    integer, intent(in) :: lrad, mpol
    real(rkind), intent(inout) :: zernike(0:lrad, 0:mpol, 0:2)

    real(rkind) :: rm, rm1, rm2 ! r to the power of m'th, m-1'th and m-2'th
    real(rkind) :: factor1, factor2, factor3, factor4
    integer :: m, n ! Zernike R^m_n

    rm = 1.0_rkind ! r to the power of m'th
    rm1 = 0.0_rkind ! r to the power of m-1'th
    rm2 = 0.0_rkind ! r to the power of m-2'th
    zernike(:, :, :) = 0.0_rkind
    do m = 0, mpol
      if (lrad >= m) then
        zernike(m, m, 0:2) = [rm, real(m, rkind) * rm1, real(m * (m - 1) , rkind) * rm2]
      end if

      if (lrad >= m + 2) then
        zernike(m + 2, m, 0) = real(m + 2, rkind) * rm * r ** 2 - real(m + 1, rkind) * rm
        zernike(m + 2, m, 1) = real( (m + 2)  ** 2, rkind) * rm * r - real( (m + 1) * m, rkind) * rm1
        zernike(m + 2, m, 2) = real( (m + 2)  ** 2 * (m + 1) , rkind) * rm - real( (m + 1) * m * (m - 1) , rkind) * rm2
      end if

      do n = m + 4, lrad, 2
        factor1 = real(n, rkind) / real(n ** 2 - m ** 2, rkind)
        factor2 = real(4 * (n - 1) , rkind)
        factor3 = real( (n - 2 + m)  ** 2, rkind) / real(n - 2, rkind) + real( (n - m)  ** 2, rkind) / real(n, rkind)
        factor4 = real( (n - 2)  ** 2 - m ** 2, rkind) / real(n - 2, rkind)

        zernike(n, m, 0) = factor1 * ( (factor2 * r ** 2 - factor3) * zernike(n - 2, m, 0) - factor4 * zernike(n - 4, m, 0))
        zernike(n, m, 1) = factor1 * (2.0_rkind * factor2 * r * zernike(n - 2, m, 0) + (factor2 * r ** 2 - factor3) *&
            & zernike(n - 2, m, 1) - factor4 * zernike(n - 4, m, 1))
        zernike(n, m, 2) = factor1 * (2.0_rkind * factor2 * (2.0_rkind * r * zernike(n - 2, m, 1) + zernike(n - 2, m, 0))&
            & + (factor2 * r ** 2 - factor3) * zernike(n - 2, m, 2) - factor4 * zernike(n - 4, m, 2))
      end do

      rm2 = rm1
      rm1 = rm
      rm = rm * r

    end do
    do n = 2, lrad, 2
      zernike(n, 0, 0) = zernike(n, 0, 0) - (-1)  **  (n / 2)
    end do
    if (mpol >= 1) then
      do n = 3, lrad, 2
        zernike(n, 1, 0) = zernike(n, 1, 0) - (-1)  **  ( (n - 1) / 2) * real( (n + 1) / 2, rkind) * r
        zernike(n, 1, 1) = zernike(n, 1, 1) - (-1)  **  ( (n - 1) / 2) * real( (n + 1) / 2, rkind)
      end do
    end if

    do m = 0, mpol
      do n = m, lrad, 2
        zernike(n, m, :) = zernike(n, m, :) / real(n + 1, rkind)
      end do
    end do
  end subroutine get_zernike_d2

  subroutine get_zernike_rm(r, lrad, mpol, zernike)
    real(rkind), intent(in) :: r
    integer, intent(in) :: lrad, mpol
    real(rkind), intent(inout) :: zernike(0:lrad, 0:mpol)

    real(rkind) :: factor1, factor2, factor3, factor4
    integer :: m, n ! Zernike R^m_n

    zernike(:, :) = 0.0_rkind
    do m = 0, mpol
      if (lrad >= m) then
        zernike(m, m) = 1.0_rkind
      end if

      if (lrad >= m + 2) then
        zernike(m + 2, m) = real(m + 2, rkind) * r ** 2 - real(m + 1, rkind)
      end if

      do n = m + 4, lrad, 2
        factor1 = real(n, rkind) / real(n ** 2 - m ** 2, rkind)
        factor2 = real(4 * (n - 1) )
        factor3 = real( (n - 2 + m)  ** 2, rkind) / real(n - 2, rkind) + real( (n - m)  ** 2, rkind) / real(n, rkind)
        factor4 = real( (n - 2)  ** 2 - m ** 2, rkind) / real(n - 2, rkind)

        zernike(n, m) = factor1 * ( (factor2 * r ** 2 - factor3) * zernike(n - 2, m) - factor4 * zernike(n - 4, m))
      end do

    end do
    do n = 2, lrad, 2
      zernike(n, 0) = zernike(n, 0) - (-1)  **  (n / 2)
    end do
    if (mpol >= 1) then
      do n = 3, lrad, 2
        zernike(n, 1) = zernike(n, 1) - (-1)  **  ( (n - 1) / 2) * real( (n + 1) / 2, rkind)
      end do
    end if

    do m = 0, mpol
      do n = m, lrad, 2
        zernike(n, m) = zernike(n, m) / real(n + 1, rkind)
      end do
    end do
  end subroutine get_zernike_rm
end module base_functions_mod
