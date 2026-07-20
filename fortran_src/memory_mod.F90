! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module memory_mod
  use memory_allocation_mod, only: sallocate, nallocate
  use io_mod
  use input_list_mod, only: Wmemory, Wmacros, enforce_stell_sym, Mpol, Lrad
  use allglobal_mod
  use cputiming_mod
  use precisions_mod, only: rkind
  use beltrami_operator_mod, only: op_select_backend, op_use_sparse_precond, BK_ITER

  implicit none(type, external)

  private
  public :: allocate_Beltrami_matrices, deallocate_Beltrami_matrices, allocate_geometry_matrices, deallocate_geometry_matrices

contains

  subroutine allocate_Beltrami_matrices(vvol, LcomputeDerivatives)
    integer, intent(in) :: vvol
    logical, intent(in) :: LcomputeDerivatives

    integer :: NN, MA

    NN = NAdof(vvol)! shorthand

    ! For the matrix-free forward solve using the flux-averaged (sparse) preconditioner the dense
    ! operator is never formed: allocate dMA/dMD at minimal size. This is the memory win at high
    ! resolution (the vacuum's dense operator is many GB). Iterative volumes that use the exact
    ! same-(m,n) preconditioner still need the transient dense operator to extract it from.
    if ((.not. LcomputeDerivatives) .and. op_select_backend(vvol) == BK_ITER &
        .and. op_use_sparse_precond(vvol)) then
      MA = 0
    else
      MA = NN
    end if

    call nallocate(dMA, 0, MA, 0, MA)! required for both plasma region and vacuum region
    call nallocate(dMD, 0, MA, 0, MA)

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
    else
      lldof = Lrad(vvol)
    end if

    ! The metric-integral tensors (DToocc, TTssss, ...) carry a mode-pair (ii,jj) index. Assembling
    ! the dense operator needs all mn^2 pairs; a matrix-free volume needs only the diagonal jj == ii,
    ! which is its exact same-(m,n) preconditioner block. Collapsing the fourth subscript takes the
    ! tensors from O(mn^2 Lrad^2) to O(mn Lrad^2) -- at m21n17 that is ~5.4 GB -> ~7 MB, which is what
    ! makes the exact block affordable for the large volumes at all. guvijsave stays full size
    ! because the matvec needs it.
    !
    ! LcomputeDerivatives is the caller's declaration of intent: .false. means "I will take the
    ! diagonal path", .true. means "I will assemble the dense operator and need the full tensors"
    ! (the wrapper self-test helpers rely on this). tensor_jslot() is the shared writer/reader
    ! resolver, and matrix() asserts it was given full-size tensors.
    if ((.not. LcomputeDerivatives) .and. op_select_backend(vvol) == BK_ITER) then
      iidof = mn
      jjdof = 1
    else
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
