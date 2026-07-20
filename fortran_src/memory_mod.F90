! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module memory_mod
  use memory_allocation_mod, only: sallocate, nallocate
  use io_mod
  use input_list_mod, only: Wmemory, Wmacros, enforce_stell_sym, Mpol, Lrad, Lfreebound, lvcvacuum
  use allglobal_mod
  use cputiming_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: allocate_Beltrami_matrices, deallocate_Beltrami_matrices, allocate_geometry_matrices, deallocate_geometry_matrices

contains

  subroutine allocate_Beltrami_matrices(vvol, LcomputeDerivatives)
    integer, intent(in) :: vvol
    logical, intent(in) :: LcomputeDerivatives

    integer :: NN

    NN = NAdof(vvol)! shorthand

    call nallocate(dMA, 0, NN, 0, NN)! required for both plasma region and vacuum region
    call nallocate(dMD, 0, NN, 0, NN)

    if (Lfreebound == 1 .and. lvcvacuum == 1 .and. vvol == Mvol) then
      call nallocate(dVC, 0, NN, 0, NN)! Laplace-virtual-casing matrix, vacuum volume only
    end if

    ! we will need the rest even with or without matrix-free
    call nallocate(dMB, 0, NN, 0, 2)
    call nallocate(dMG, 0, NN)

    call sallocate(solution, 1, NN, -1, 2, 0.0_rkind)! this will contain the vector potential from the linear solver and
    ! its derivatives

  end subroutine allocate_Beltrami_matrices

  subroutine deallocate_Beltrami_matrices(LcomputeDerivatives)
    logical, intent(in) :: LcomputeDerivatives

    deallocate(dMA)
    deallocate(dMD)

    if (allocated(dVC)) deallocate(dVC)

    deallocate(dMB)

    deallocate(dMG)

    deallocate(solution)

  end subroutine deallocate_Beltrami_matrices

  subroutine allocate_geometry_matrices(vvol, LcomputeDerivatives, ntz_loc)
    integer, intent(in) :: vvol, ntz_loc
    logical, intent(in) :: LcomputeDerivatives

    integer :: ll, lldof, jjdof, iidof

    ll = Lrad(vvol)

    if (Lcoordinatesingularity) then ! different radial dof for Zernike
      lldof = (Lrad(vvol) - mod(Lrad(vvol), 2)) / 2

      ! we need full-size matrices
      iidof = mn
      jjdof = mn
    else
      lldof = Lrad(vvol)
      iidof = mn
      jjdof = mn
    end if

    call nallocate(guvijsave, 1, ntz_loc, 1, 3, 1, 3, 1, Iquad(vvol))

    call nallocate(DToocc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(TTssss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(TDstsc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(TDszsc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(DDttcc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(DDtzcc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
    call nallocate(DDzzcc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

    call nallocate(Tss, 0, lldof, 1, mn)
    call nallocate(Dtc, 0, lldof, 1, mn)
    call nallocate(Dzc, 0, lldof, 1, mn)
    call nallocate(Ttc, 0, lldof, 1, mn)
    call nallocate(Tzc, 0, lldof, 1, mn)

    if (.not.enforce_stell_sym) then

      call nallocate(DToocs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DToosc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DTooss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(TTsscc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TTsscs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TTsssc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(TDstcc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TDstcs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TDstss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(TDszcc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TDszcs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(TDszss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(DDttcs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDttsc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDttss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(DDtzcs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDtzsc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDtzss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(DDzzcs, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDzzsc, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)
      call nallocate(DDzzss, 0, lldof, 0, lldof, 1, iidof, 1, jjdof)

      call nallocate(Tsc, 0, lldof, 1, mn)
      call nallocate(Dts, 0, lldof, 1, mn)
      call nallocate(Dzs, 0, lldof, 1, mn)
      call nallocate(Tts, 0, lldof, 1, mn)
      call nallocate(Tzs, 0, lldof, 1, mn)

    end if

  end subroutine allocate_geometry_matrices

  subroutine deallocate_geometry_matrices(LcomputeDerivatives)
    logical, intent(in) :: LcomputeDerivatives

    Lsavedguvij = .false.
    deallocate(guvijsave)

    deallocate(DToocc)
    deallocate(TTssss)
    deallocate(TDstsc)
    deallocate(TDszsc)
    deallocate(DDttcc)
    deallocate(DDtzcc)
    deallocate(DDzzcc)

    deallocate(Tss)
    deallocate(Dtc)
    deallocate(Dzc)
    deallocate(Ttc)
    deallocate(Tzc)

    if (.not.enforce_stell_sym) then
      deallocate(DToocs)
      deallocate(DToosc)
      deallocate(DTooss)

      deallocate(TTsscc)
      deallocate(TTsscs)
      deallocate(TTsssc)

      deallocate(TDstcc)
      deallocate(TDstcs)
      deallocate(TDstss)

      deallocate(TDszcc)
      deallocate(TDszcs)
      deallocate(TDszss)

      deallocate(DDttcs)
      deallocate(DDttsc)
      deallocate(DDttss)

      deallocate(DDtzcs)
      deallocate(DDtzsc)
      deallocate(DDtzss)

      deallocate(DDzzcs)
      deallocate(DDzzsc)
      deallocate(DDzzss)

      deallocate(Tsc)
      deallocate(Dts)
      deallocate(Dzs)
      deallocate(Tts)
      deallocate(Tzs)
    end if
  end subroutine deallocate_geometry_matrices
end module memory_mod
