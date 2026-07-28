! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> Owns the process-local state used only while evaluating force Jacobian-vector products.
!>
!> The legacy geometry call chain cannot accept an explicit context without changing the protected
!> dense Jacobian path. This module therefore owns one serial context. Callers reserve its buffers
!> before entering a direction batch, activate it for one volume and angular grid at a time, and
!> borrow zero-copy views only for the duration of that activation. The status remains available
!> after the work buffers are released so the Python boundary can consume it.
module force_jvp_context_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private

  public :: force_jvp_scope
  public :: finalize_jvp_scope
  public :: jvp_context_activate, jvp_context_borrow_axis, jvp_context_borrow_direction
  public :: jvp_context_clear_axis, jvp_context_close, jvp_context_identity
  public :: jvp_context_initialize, jvp_context_is_active, jvp_context_open
  public :: jvp_context_raise_status, jvp_context_release_work, jvp_context_reserve
  public :: jvp_context_reset, jvp_context_reset_status, jvp_context_set_all_status
  public :: jvp_context_status, jvp_context_status_size

  !> Scope guard for the work buffers and active flag of the serial JVP context.
  type :: force_jvp_scope
    private
    logical :: owns_context = .false.
  contains
    final :: finalize_jvp_scope
  end type force_jvp_scope

  logical, save :: active = .false.
  integer, save :: active_grid_size = 0
  integer, save :: active_nmode = 0
  integer, save :: active_volume = 0

  real(rkind), allocatable, target, save :: direction_rc(:, :)
  real(rkind), allocatable, target, save :: direction_rs(:, :)
  real(rkind), allocatable, target, save :: direction_zc(:, :)
  real(rkind), allocatable, target, save :: direction_zs(:, :)
  real(rkind), allocatable, target, save :: axis_r(:)
  real(rkind), allocatable, target, save :: axis_rz(:)
  real(rkind), allocatable, target, save :: axis_z(:)
  real(rkind), allocatable, target, save :: axis_zz(:)
  integer, allocatable, save :: status(:)

contains

  !> Initialize persistent diagnostics for a new SPECTRE configuration.
  subroutine jvp_context_initialize(nvolume)
    integer, intent(in) :: nvolume

    if (allocated(status)) then
      if (size(status) /= nvolume) deallocate(status)
    end if
    if (.not. allocated(status)) allocate(status(1:nvolume))
    status = 0
    call jvp_context_release_work()
  end subroutine jvp_context_initialize

  !> Begin an owned JVP scope. Its finalizer releases work storage on every return path.
  subroutine jvp_context_open(scope)
    type(force_jvp_scope), intent(out) :: scope

    call jvp_context_release_work()
    scope%owns_context = .true.
  end subroutine jvp_context_open

  !> End an owned JVP scope explicitly, retaining only the most recent status diagnostics.
  subroutine jvp_context_close(scope)
    type(force_jvp_scope), intent(inout) :: scope

    if (.not. scope%owns_context) return
    call jvp_context_release_work()
    scope%owns_context = .false.
  end subroutine jvp_context_close

  subroutine finalize_jvp_scope(scope)
    type(force_jvp_scope), intent(inout) :: scope

    call jvp_context_close(scope)
  end subroutine finalize_jvp_scope

  !> Reserve all direction and axis buffers before entering the JVP direction loop.
  subroutine jvp_context_reserve(nmode, max_grid_size)
    integer, intent(in) :: nmode, max_grid_size

    if (allocated(direction_rc)) then
      if (size(direction_rc, 1) /= nmode) then
        deallocate(direction_rc, direction_rs, direction_zc, direction_zs)
      end if
    end if
    if (.not. allocated(direction_rc)) then
      allocate(direction_rc(1:nmode, 0:1), direction_rs(1:nmode, 0:1))
      allocate(direction_zc(1:nmode, 0:1), direction_zs(1:nmode, 0:1))
    end if

    if (allocated(axis_r)) then
      if (size(axis_r) < max_grid_size) deallocate(axis_r, axis_rz, axis_z, axis_zz)
    end if
    if (.not. allocated(axis_r)) then
      allocate(axis_r(1:max_grid_size), axis_rz(1:max_grid_size))
      allocate(axis_z(1:max_grid_size), axis_zz(1:max_grid_size))
    end if
  end subroutine jvp_context_reserve

  !> Activate zeroed direction coefficients for one volume and angular grid.
  subroutine jvp_context_activate(nmode, grid_size, volume)
    integer, intent(in) :: nmode, grid_size, volume

    call jvp_context_reserve(nmode, grid_size)
    direction_rc(1:nmode, 0:1) = 0.0_rkind
    direction_rs(1:nmode, 0:1) = 0.0_rkind
    direction_zc(1:nmode, 0:1) = 0.0_rkind
    direction_zs(1:nmode, 0:1) = 0.0_rkind
    active = .true.
    active_nmode = nmode
    active_grid_size = grid_size
    active_volume = volume
  end subroutine jvp_context_activate

  !> Borrow the current direction coefficient tables without allocating or copying.
  subroutine jvp_context_borrow_direction(rc, rs, zc, zs)
    real(rkind), pointer, contiguous, intent(out) :: rc(:, :), rs(:, :), zc(:, :), zs(:, :)

    rc => direction_rc
    rs => direction_rs
    zc => direction_zc
    zs => direction_zs
  end subroutine jvp_context_borrow_direction

  !> Zero the axis response on the active grid before its owner assembles it.
  subroutine jvp_context_clear_axis()
    axis_r(1:active_grid_size) = 0.0_rkind
    axis_rz(1:active_grid_size) = 0.0_rkind
    axis_z(1:active_grid_size) = 0.0_rkind
    axis_zz(1:active_grid_size) = 0.0_rkind
  end subroutine jvp_context_clear_axis

  !> Borrow the current grid's axis-response buffers without allocating or copying.
  subroutine jvp_context_borrow_axis(r, rz, z, zz)
    real(rkind), pointer, contiguous, intent(out) :: r(:), rz(:), z(:), zz(:)

    r => axis_r(1:active_grid_size)
    rz => axis_rz(1:active_grid_size)
    z => axis_z(1:active_grid_size)
    zz => axis_zz(1:active_grid_size)
  end subroutine jvp_context_borrow_axis

  !> Return whether general-direction geometry is active.
  logical function jvp_context_is_active() result(is_active)
    is_active = active
  end function jvp_context_is_active

  !> Return the identity of the currently loaded direction payload.
  subroutine jvp_context_identity(volume, nmode, grid_size)
    integer, intent(out) :: volume, nmode, grid_size

    volume = active_volume
    nmode = active_nmode
    grid_size = active_grid_size
  end subroutine jvp_context_identity

  !> Deactivate the context while retaining its reusable buffers and diagnostics.
  subroutine jvp_context_reset()
    active = .false.
    active_volume = 0
    active_nmode = 0
    active_grid_size = 0
  end subroutine jvp_context_reset

  !> Deactivate the context and release all direction and axis buffers.
  subroutine jvp_context_release_work()
    call jvp_context_reset()
    if (allocated(direction_rc)) deallocate(direction_rc, direction_rs, direction_zc, direction_zs)
    if (allocated(axis_r)) deallocate(axis_r, axis_rz, axis_z, axis_zz)
  end subroutine jvp_context_release_work

  !> Clear the per-volume result status before a JVP batch.
  subroutine jvp_context_reset_status()
    if (allocated(status)) status = 0
  end subroutine jvp_context_reset_status

  !> Set every per-volume status, used for unsupported global configurations.
  subroutine jvp_context_set_all_status(value)
    integer, intent(in) :: value

    if (allocated(status)) status = value
  end subroutine jvp_context_set_all_status

  !> Raise one per-volume status while retaining the most severe existing positive code.
  subroutine jvp_context_raise_status(volume, value)
    integer, intent(in) :: volume, value

    if (allocated(status)) status(volume) = max(status(volume), value)
  end subroutine jvp_context_raise_status

  !> Return one per-volume status value for compatibility adapters and diagnostics.
  integer function jvp_context_status(volume) result(value)
    integer, intent(in) :: volume

    value = status(volume)
  end function jvp_context_status

  !> Return the number of per-volume status entries.
  integer function jvp_context_status_size() result(nvolume)
    if (allocated(status)) then
      nvolume = size(status)
    else
      nvolume = 0
    end if
  end function jvp_context_status_size

end module force_jvp_context_mod
