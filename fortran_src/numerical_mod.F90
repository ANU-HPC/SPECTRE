! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

module numerical_mod
  use precisions_mod, only: rkind

  implicit none

  private
  public :: gi00ab, getimn, tfft, invfft, gauleg

contains

  subroutine gi00ab(Mpol, Ntor, Nfp, mn, im, in)

    implicit none

    integer, intent(in) :: Mpol, Ntor, Nfp, mn
    integer, intent(out) :: im(mn), in(mn)

    integer :: imn, mm, nn

    imn = 0

    mm = 0
    do nn = 0, Ntor
      imn = imn + 1
      im(imn) = mm
      in(imn) = nn * Nfp
    end do

    do mm = 1, Mpol
      do nn = -Ntor, Ntor
        imn = imn + 1
        im(imn) = mm
        in(imn) = nn * Nfp
      end do
    end do

    return

  end subroutine gi00ab

  subroutine getimn(Mpol, Ntor, Nfp, mi, ni, idx)
    ! convert m and n to index
    implicit none
    integer, intent(in) :: Mpol, Ntor, Nfp, mi, ni
    integer, intent(out) :: idx

    if (mi > Mpol .or. mi < 0 .or. ni > Ntor * Nfp .or. ni < -Ntor * Nfp) then
      idx = 0
    elseif (mi == 0) then
      idx = 1 + ni / Nfp
    else
      idx = 1 + Ntor + (2 * Ntor + 1) * (mi - 1) + (ni / Nfp + Ntor + 1)
    end if

  end subroutine getimn

  subroutine tfft(ijreal, ijimag, mn, im, in, efmn, ofmn, cfmn, sfmn, ifail)
    use precisions_mod, only: rkind
    use constants_mod, only: pi2
    use input_list_mod, only: Nfp
    use allglobal_mod, only: pi2nfp
    use fft_backend_mod, only: nt_plan, nz_plan, cplxin, cplxout, fft_forward, c_double_complex

    implicit none

    intrinsic aimag

    integer :: mn, im(1:mn), in(1:mn), ntz_plan, imn, ifail, mm, nn
    real(rkind) :: ijreal(1:nt_plan * nz_plan), ijimag(1:nt_plan * nz_plan), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn)

    logical :: Lcheck = .false.
    integer :: jj, kk
    real(rkind) :: arg, ca, sa
    complex(c_double_complex) :: z1, z2, z3

    do jj = 1, nz_plan
      cplxin(:, jj) = cmplx(ijreal( (jj - 1) * nt_plan + 1:jj * nt_plan), ijimag( (jj - 1) * nt_plan + 1:jj * nt_plan), KIND=c_double_complex)
    end do

    call fft_forward()!Forward transform
    ntz_plan = nt_plan * nz_plan
    cplxout(:, :) = cplxout(:, :) / ntz_plan
    cplxout(1, 1) = 1.0_rkind / 2.0_rkind * cplxout(1, 1)

    do imn = 1, mn
      mm = im(imn)
      nn = in(imn) / Nfp

      z1 = cplxout(1 + mod(nt_plan - mm, nt_plan), 1 + mod(nz_plan + nn, nz_plan))
      z2 = cplxout(1 + mm, 1 + mod(nz_plan - nn, nz_plan))

      z3 = z1 + z2
      efmn(imn) = real(z3, rkind)
      cfmn(imn) = aimag(z3)

      z3 = z1 - z2
      ofmn(imn) = aimag(z3)
      sfmn(imn) = -real(z3, rkind)
    end do

    if (.not.Lcheck) return

    ijreal(1:ntz_plan) = 0.0_rkind
    ijimag(1:ntz_plan) = 0.0_rkind

    do jj = 0, nt_plan - 1

      do kk = 0, nz_plan - 1

        do imn = 1, mn
          arg = im(imn) * jj * pi2 / nt_plan - in(imn) * kk * pi2nfp / nz_plan
          ca = cos(arg)
          sa = sin(arg)

          ijreal(1 + jj + kk * nt_plan) = ijreal(1 + jj + kk * nt_plan) + efmn(imn) * ca + ofmn(imn) * sa
          ijimag(1 + jj + kk * nt_plan) = ijimag(1 + jj + kk * nt_plan) + cfmn(imn) * ca + sfmn(imn) * sa

        end do
      end do
    end do
  end subroutine tfft

  subroutine invfft(mn, im, in, efmn, ofmn, cfmn, sfmn, ijreal, ijimag)
    use precisions_mod, only: rkind
    use input_list_mod, only: Nfp
    use fft_backend_mod, only: nt_plan, nz_plan, cplxin, cplxout, fft_backward, c_double_complex

    implicit none

    integer, intent(in) :: mn, im(mn), in(mn)
    real(rkind), intent(in) :: efmn(mn), ofmn(mn), cfmn(mn), sfmn(mn)
    real(rkind), intent(out) :: ijreal(nt_plan * nz_plan), ijimag(nt_plan * nz_plan)! output real space

    integer :: imn, jj, mm, nn

    cplxin(:, :) = 0.0_rkind

    !Copy real arrays to complex
    do imn = 1, mn
      mm = im(imn)
      nn = in(imn) / Nfp
      cplxin(1 + mod(nt_plan - mm, nt_plan), 1 + mod(nz_plan + nn, nz_plan)) = 1.0_rkind / 2.0_rkind * cmplx(efmn(imn) - sfmn(imn), cfmn(imn)&
          & + ofmn(imn), KIND=c_double_complex)
      cplxin(1 + mm, 1 + mod(nz_plan - nn, nz_plan)) = 1.0_rkind / 2.0_rkind * cmplx(efmn(imn) + sfmn(imn), cfmn(imn) - ofmn(imn),&
          & KIND=c_double_complex)
    end do
    cplxin(1, 1) = 2.0_rkind * cplxin(1, 1)

    call fft_backward()!Inverse transform

    !Copy complex result back to real arrays
    do jj = 1, nz_plan
      ijreal( (jj - 1) * nt_plan + 1:jj * nt_plan) = real(cplxout(:, jj), rkind)
      ijimag( (jj - 1) * nt_plan + 1:jj * nt_plan) = aimag(cplxout(:, jj))
    end do

    return

  end subroutine invfft

  subroutine gauleg(n, weight, abscis, ifail)
    use precisions_mod, only: rkind
    use constants_mod, only: pi

    implicit none

    intrinsic abs, cos, epsilon

    integer, intent(in) :: n
    real(rkind), dimension(n), intent(out) :: weight, abscis
    integer, intent(out) :: ifail

    integer, parameter :: maxiter = 16
    integer :: m, j, i, irefl, iter
    real(rkind) :: z1, z, pp, p3, p2, p1
    real(rkind), parameter :: eps = epsilon(z)

    !Error checking
    if (n < 1) then
      ifail = 2
      return
    end if

    m = (n + 1) / 2 !Roots are symmetric in interval, so we only need 1.0_rkind / 2.0_rkind
    do i = 1, m !Loop over desired roots
      irefl = n + 1 - i
      if (i /= irefl) then
        z = cos(pi * (i - 0.25_rkind) / (n + 0.5_rkind) )! Approximate ith root
      else !For an odd number of abscissae, the center must be at 0.0_rkind by symmetry.
        z = 0.0_rkind
      end if

      !Refine by Newton method
      do iter = 1, maxiter
        p1 = 1.0_rkind
        p2 = 0.0_rkind ! Initialize recurrence relation

        do j = 1, n !Recurrence relation to get P(x)
          p3 = p2
          p2 = p1
          p1 = ( (2.0_rkind * j - 1.0_rkind) * z * p2 - (j - 1.0_rkind) * p3) / j
        end do !j

        pp = n * (z * p1 - p2) / (z * z - 1.0_rkind) !Derivative of P(x)
        z1 = z
        z = z1 - p1 / pp !Newton iteration
        if (abs(z - z1) <= eps) exit !Convergence test
      end do !iter
      if (iter > maxiter) then
        ifail = 1
        return
      end if

      abscis(i) = -z
      abscis(irefl) = z
      weight(i) = 2.0_rkind / ( (1.0_rkind - z * z) * pp * pp)
      weight(irefl) = weight(i)
    end do !i

    ifail = 0
  end subroutine gauleg

end module numerical_mod
