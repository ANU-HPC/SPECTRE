! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module perturbed_workspace_mod
  use precisions_mod, only: rkind

  implicit none

  private
  public :: perturbed_rhs_workspace, alloc_perturbed_workspace, dealloc_perturbed_workspace

  type :: perturbed_rhs_workspace
    ! Holds only the large, hot per-thread scratch arrays. The per-DOF descriptor (dBdX) and the
    ! flags (Lsavedguvij, region flags) are NOT stored here: they are small scalars set per DOF in
    ! the driver loop and read deep in the call chain, and they remain plain module globals (this
    ! DOF loop is serial). Keeping them out of this struct keeps it purely the large-array scratch.

    ! intghs workspace (mirrors intghs_workspace fields in intghs_mod)
    real(rkind), allocatable :: gBupper(:, :, :)
    real(rkind), allocatable :: Blower(:, :, :)
    real(rkind), allocatable :: Bloweremn(:, :, :)
    real(rkind), allocatable :: Bloweromn(:, :, :)
    real(rkind), allocatable :: efmn(:, :)
    real(rkind), allocatable :: odmn(:, :)
    real(rkind), allocatable :: cfmn(:, :)
    real(rkind), allocatable :: ijreal(:, :)
    real(rkind), allocatable :: jkreal(:, :)
    real(rkind), allocatable :: basis(:, :, :, :)
    real(rkind), allocatable :: ofmn(:, :)
    real(rkind), allocatable :: evmn(:, :)
    real(rkind), allocatable :: sfmn(:, :)
    real(rkind), allocatable :: kjreal(:, :)
    real(rkind), allocatable :: jireal(:, :)

    ! metric/geometry scratch (mirrors allglobal guvijsave, guvij, Rij, Zij, sg, inv_jac)
    real(rkind), allocatable :: guvijsave(:, :, :, :)  ! (Ntz, 1:3, 1:3, max_iquad)
    real(rkind), allocatable :: guvij(:, :, :, :)       ! (Ntz, 0:3, 0:3, -1:3)
    real(rkind), allocatable :: Rij(:, :, :)            ! (Ntz, 0:3, 0:3)
    real(rkind), allocatable :: Zij(:, :, :)            ! (Ntz, 0:3, 0:3)
    real(rkind), allocatable :: sg(:, :)                ! (Ntz, 0:3)
    real(rkind), allocatable :: inv_jac(:)              ! (Ntz)

    ! projection scratch (mirrors allglobal Tss/Dtc/Dzc/Tsc/Dts/Dzs)
    real(rkind), allocatable :: Tss(:, :)
    real(rkind), allocatable :: Dtc(:, :)
    real(rkind), allocatable :: Dzc(:, :)
    real(rkind), allocatable :: Tsc(:, :)
    real(rkind), allocatable :: Dts(:, :)
    real(rkind), allocatable :: Dzs(:, :)
  end type perturbed_rhs_workspace

contains

  subroutine alloc_perturbed_workspace(pws, Ntz, mn, max_iquad, max_lrad, mpol, max_lldof)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    integer, intent(in) :: Ntz, mn, max_iquad, max_lrad, mpol, max_lldof

    allocate(pws%gBupper(1:Ntz, 1:3, 1:max_iquad))
    allocate(pws%Blower(1:Ntz, 1:3, 1:max_iquad))
    allocate(pws%Bloweremn(1:mn, 1:3, 1:max_iquad))
    allocate(pws%Bloweromn(1:mn, 1:3, 1:max_iquad))
    allocate(pws%efmn(1:mn, 1:max_iquad))
    allocate(pws%odmn(1:mn, 1:max_iquad))
    allocate(pws%cfmn(1:mn, 1:max_iquad))
    allocate(pws%ijreal(1:mn, 1:max_iquad))
    allocate(pws%jkreal(1:mn, 1:max_iquad))
    allocate(pws%basis(0:max_lrad, 0:mpol, 0:1, 1:max_iquad))
    allocate(pws%ofmn(1:mn, 1:max_iquad))
    allocate(pws%evmn(1:mn, 1:max_iquad))
    allocate(pws%sfmn(1:mn, 1:max_iquad))
    allocate(pws%kjreal(1:mn, 1:max_iquad))
    allocate(pws%jireal(1:mn, 1:max_iquad))

    allocate(pws%guvijsave(1:Ntz, 1:3, 1:3, 1:max_iquad))
    allocate(pws%guvij(1:Ntz, 0:3, 0:3, -1:3))
    allocate(pws%Rij(1:Ntz, 0:3, 0:3))
    allocate(pws%Zij(1:Ntz, 0:3, 0:3))
    allocate(pws%sg(1:Ntz, 0:3))
    allocate(pws%inv_jac(1:Ntz))

    allocate(pws%Tss(0:max_lldof, 1:mn))
    allocate(pws%Dtc(0:max_lldof, 1:mn))
    allocate(pws%Dzc(0:max_lldof, 1:mn))
    allocate(pws%Tsc(0:max_lldof, 1:mn))
    allocate(pws%Dts(0:max_lldof, 1:mn))
    allocate(pws%Dzs(0:max_lldof, 1:mn))
  end subroutine alloc_perturbed_workspace

  subroutine dealloc_perturbed_workspace(pws)
    type(perturbed_rhs_workspace), intent(inout) :: pws

    if (allocated(pws%gBupper))    deallocate(pws%gBupper)
    if (allocated(pws%Blower))     deallocate(pws%Blower)
    if (allocated(pws%Bloweremn))  deallocate(pws%Bloweremn)
    if (allocated(pws%Bloweromn))  deallocate(pws%Bloweromn)
    if (allocated(pws%efmn))       deallocate(pws%efmn)
    if (allocated(pws%odmn))       deallocate(pws%odmn)
    if (allocated(pws%cfmn))       deallocate(pws%cfmn)
    if (allocated(pws%ijreal))     deallocate(pws%ijreal)
    if (allocated(pws%jkreal))     deallocate(pws%jkreal)
    if (allocated(pws%basis))      deallocate(pws%basis)
    if (allocated(pws%ofmn))       deallocate(pws%ofmn)
    if (allocated(pws%evmn))       deallocate(pws%evmn)
    if (allocated(pws%sfmn))       deallocate(pws%sfmn)
    if (allocated(pws%kjreal))     deallocate(pws%kjreal)
    if (allocated(pws%jireal))     deallocate(pws%jireal)
    if (allocated(pws%guvijsave))  deallocate(pws%guvijsave)
    if (allocated(pws%guvij))      deallocate(pws%guvij)
    if (allocated(pws%Rij))        deallocate(pws%Rij)
    if (allocated(pws%Zij))        deallocate(pws%Zij)
    if (allocated(pws%sg))         deallocate(pws%sg)
    if (allocated(pws%inv_jac))    deallocate(pws%inv_jac)
    if (allocated(pws%Tss))        deallocate(pws%Tss)
    if (allocated(pws%Dtc))        deallocate(pws%Dtc)
    if (allocated(pws%Dzc))        deallocate(pws%Dzc)
    if (allocated(pws%Tsc))        deallocate(pws%Tsc)
    if (allocated(pws%Dts))        deallocate(pws%Dts)
    if (allocated(pws%Dzs))        deallocate(pws%Dzs)
  end subroutine dealloc_perturbed_workspace

end module perturbed_workspace_mod
