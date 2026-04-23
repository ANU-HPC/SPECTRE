! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module metric_mod
  use precisions_mod, only: rkind
  use input_list_mod, only: Wcompute_metric_quantities
  use allglobal_mod, only: mne, ime, ine, Nt, Nz, Ntz, cfmn, sfmn, ijreal, gvuij, goomne, goomno, gssmne, gssmno, gstmne, &
      gstmno, gszmne, gszmno, gttmne, gttmno, gtzmne, gtzmno, gzzmne, gzzmno, guvijsave, gaussianabscissae, &
      mn, guvij, sg, inv_jac
  use numerical_mod, only: tfft
  use geometry_mod, only: compute_coordinates

  implicit none(type, external)

  private
  public :: compute_metric_quantities, compute_guvijsave

contains

  subroutine compute_metric_quantities(lquad, lvol, ntz_loc)
    integer, intent(in) :: lvol, lquad, ntz_loc

    integer :: ifail, jquad

    do jquad = 1, lquad
      gvuij(1:ntz_loc, 0, 0) = 1.0_rkind

      gvuij(1:ntz_loc, 1, 1) = guvijsave(1:ntz_loc, 1, 1, jquad)
      gvuij(1:ntz_loc, 1, 2) = guvijsave(1:ntz_loc, 1, 2, jquad)
      gvuij(1:ntz_loc, 1, 3) = guvijsave(1:ntz_loc, 1, 3, jquad)
      gvuij(1:ntz_loc, 2, 2) = guvijsave(1:ntz_loc, 2, 2, jquad)
      gvuij(1:ntz_loc, 2, 3) = guvijsave(1:ntz_loc, 2, 3, jquad)
      gvuij(1:ntz_loc, 3, 3) = guvijsave(1:ntz_loc, 3, 3, jquad)

      ifail = 0
      call tfft(gvuij(1:ntz_loc, 0, 0), ijreal(1:ntz_loc), mne, ime(1:mne), ine(1:mne), goomne(1:mne, jquad), &
          goomno(1:mne, jquad), cfmn(1:mne), sfmn(1:mne), ifail)
      goomne(0, jquad) = 0.0_rkind
      goomno(0, jquad) = 0.0_rkind

      ifail = 0
      call tfft(gvuij(1:ntz_loc, 1, 1), gvuij(1:ntz_loc, 1, 2), mne, ime(1:mne), ine(1:mne), gssmne(1:mne, jquad), &
          gssmno(1:mne, jquad), gstmne(1:mne, jquad), gstmno(1:mne, jquad), ifail)
      gssmne(0, jquad) = 0.0_rkind
      gssmno(0, jquad) = 0.0_rkind
      gstmne(0, jquad) = 0.0_rkind
      gstmno(0, jquad) = 0.0_rkind

      ifail = 0
      call tfft(gvuij(1:ntz_loc, 1, 3), gvuij(1:ntz_loc, 2, 2), mne, ime(1:mne), ine(1:mne), gszmne(1:mne, jquad), &
          gszmno(1:mne, jquad), gttmne(1:mne, jquad), gttmno(1:mne, jquad), ifail)
      gszmne(0, jquad) = 0.0_rkind
      gszmno(0, jquad) = 0.0_rkind
      gttmne(0, jquad) = 0.0_rkind
      gttmno(0, jquad) = 0.0_rkind

      ifail = 0
      call tfft(gvuij(1:ntz_loc, 2, 3), gvuij(1:ntz_loc, 3, 3), mne, ime(1:mne), ine(1:mne), gtzmne(1:mne, jquad), &
          gtzmno(1:mne, jquad), gzzmne(1:mne, jquad), gzzmno(1:mne, jquad), ifail)
      gtzmne(0, jquad) = 0.0_rkind
      gtzmno(0, jquad) = 0.0_rkind
      gzzmne(0, jquad) = 0.0_rkind
      gzzmno(0, jquad) = 0.0_rkind
    end do
  end subroutine compute_metric_quantities

  subroutine compute_guvijsave(lquad, vvol, ideriv, Lcurvature, nt_loc, nz_loc)
    integer, intent(in) :: vvol, lquad, ideriv, Lcurvature, nt_loc, nz_loc
    integer :: jquad, ii, jj
    real(rkind) :: lss

    ! we need to compute guvij and save it in guvijsave
    do jquad = 1, lquad
      lss = gaussianabscissae(jquad, vvol)
      call compute_coordinates(vvol, lss, Lcurvature, nt_loc, nz_loc, mn)
      guvijsave(1:(nt_loc*nz_loc), 1:3, 1:3, jquad) = guvij(1:(nt_loc*nz_loc), 1:3, 1:3, ideriv)
      do ii = 1, 3
        do jj = 1, 3
          guvijsave(1:(nt_loc*nz_loc), jj, ii, jquad) = guvijsave(1:(nt_loc*nz_loc), jj, ii, jquad) * inv_jac(1:(nt_loc*nz_loc))
        end do
      end do
    end do
  end subroutine compute_guvijsave
end module metric_mod
