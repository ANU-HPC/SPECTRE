! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> Non-numerical ownership checks for the opaque Beltrami operator and cache manager.
module beltrami_operator_test_support_mod

  use precisions_mod, only: rkind
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_bound_to, op_is_empty, &
      op_move, op_release
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
    integer :: capacity, hits, insertions, invalidations, misses, occupied
    logical :: hit
    real(rkind), parameter :: test_mu = 1.25_rkind

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
  end subroutine operator_cache_lifecycle_selftest

end module beltrami_operator_test_support_mod
