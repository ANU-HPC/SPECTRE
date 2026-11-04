! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> Non-numerical ownership checks for the opaque Beltrami operator and cache manager.
module beltrami_operator_test_support_mod

  use precisions_mod, only: rkind
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_bind_iter_identity, &
      op_bound_to, op_has_row_correction, op_is_empty, op_move, op_release, &
      op_set_row_correction
  use beltrami_factor_cache_mod, only: factor_cache_manager

  implicit none(type, external)

  private
  public :: operator_cache_lifecycle_selftest

contains

  !> Exercise move, acquire, return, mismatch, invalidation, disabled, and clear semantics.
  subroutine operator_cache_lifecycle_selftest(status)
    integer, intent(out) :: status

    type(beltrami_operator) :: acquired, owner
    type(factor_cache_manager) :: cache
    integer :: capacity, hits, insertions, invalidations, misses, occupied, validation_status
    integer :: empty_rows(0)
    logical :: hit
    real(rkind), parameter :: test_mu = 1.25_rkind
    real(rkind) :: correction(1, 3)
    real(rkind) :: empty_values(0, 3)

    status = 0

    call op_alloc(owner, 2, 3, .false.)
    call op_move(owner, acquired)
    if (.not. op_is_empty(owner)) status = status + 1
    if (.not. op_bound_to(acquired, 2, 3)) status = status + 2

    call cache%insert(.true., 2, 2, test_mu, acquired)
    if (.not. op_is_empty(acquired)) status = status + 4
    call cache%diagnostics(capacity, occupied, hits, misses, insertions, invalidations)
    if (capacity /= 2 .or. occupied /= 1 .or. insertions /= 1) status = status + 8

    hit = cache%acquire(.true., 2, test_mu, 3, acquired)
    if (.not. hit .or. .not. op_bound_to(acquired, 2, 3)) status = status + 16
    call cache%insert(.true., 2, 2, test_mu, acquired)
    hit = cache%acquire(.true., 2, test_mu, 4, acquired)
    if (hit .or. .not. op_is_empty(acquired)) status = status + 32

    call op_alloc(owner, 1, 2, .false.)
    call cache%insert(.true., 2, 1, test_mu, owner)
    call cache%invalidate(1)
    hit = cache%acquire(.true., 1, test_mu, 2, owner)
    if (hit .or. .not. op_is_empty(owner)) status = status + 64

    call op_alloc(owner, 1, 2, .false.)
    call cache%insert(.false., 2, 1, test_mu, owner)
    if (.not. op_bound_to(owner, 1, 2)) status = status + 128
    call op_release(owner)

    call cache%clear()
    call cache%diagnostics(capacity, occupied, hits, misses, insertions, invalidations)
    if (capacity /= 0 .or. occupied /= 0) status = status + 256
    if (hits /= 1 .or. misses /= 2 .or. insertions /= 3 .or. invalidations /= 1) then
      status = status + 512
    end if

    call op_release(acquired)

    ! The correction is part of the move-only handle and therefore follows the same cache path.
    correction(1, :) = [0.5_rkind, 0.0_rkind, -1.0_rkind]
    call op_alloc(owner, 2, 3, .false.)
    call op_set_row_correction(owner, 1, [2], correction)
    if (.not. op_has_row_correction(owner)) status = status + 1024
    call op_move(owner, acquired)
    if (op_has_row_correction(owner) .or. .not. op_has_row_correction(acquired)) status = status + 2048
    call cache%insert(.true., 2, 2, test_mu, acquired)
    hit = cache%acquire(.true., 2, test_mu, 3, acquired)
    if (.not. hit .or. .not. op_has_row_correction(acquired)) status = status + 4096

    ! Release and every rebind clear the complete previous payload.
    call op_release(acquired)
    if (op_has_row_correction(acquired) .or. .not. op_is_empty(acquired)) status = status + 8192
    call op_alloc(acquired, 2, 3, .false.)
    call op_set_row_correction(acquired, 1, [2], correction)
    call op_bind_iter_identity(acquired, 2, 3)
    if (op_has_row_correction(acquired)) status = status + 16384
    call op_release(acquired)

    ! Tests use the checked form; production callers omit status and receive fatal validation.
    call op_set_row_correction(owner, 1, [1], reshape([1.0_rkind], [1, 1]), validation_status)
    if (validation_status /= 1) status = status + 32768
    call op_alloc(owner, 1, 3, .false.)
    call op_set_row_correction(owner, 1, [0], correction, validation_status)
    if (validation_status /= 3 .or. op_has_row_correction(owner)) status = status + 65536
    call op_set_row_correction(owner, 1, [4], correction, validation_status)
    if (validation_status /= 3 .or. op_has_row_correction(owner)) status = status + 131072
    call op_set_row_correction(owner, 0, empty_rows, empty_values, validation_status)
    if (validation_status /= 2 .or. op_has_row_correction(owner)) status = status + 262144
    call op_set_row_correction(owner, 2, [1], correction, validation_status)
    if (validation_status /= 4 .or. op_has_row_correction(owner)) status = status + 524288
    call op_release(owner)
    call cache%clear()
  end subroutine operator_cache_lifecycle_selftest

end module beltrami_operator_test_support_mod
