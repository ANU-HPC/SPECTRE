! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> \brief Move-only cache manager for per-volume Beltrami operator factors.
!>
!> The process cache is opt-in through Lbeltrami_factor_cache. Its explicit key is
!> (volume, NAdof, mu, geometry generation). The force evaluation invalidates a volume before
!> rebuilding its field, which advances that volume's generation; a factor inserted afterward is
!> therefore valid only for that geometry. Each rank owns its own manager and the only consumer,
!> the JVP path, is single-rank.
module beltrami_factor_cache_mod

  use precisions_mod, only: rkind
  use allglobal_mod, only: Lbeltrami_factor_cache, Mvol
  use beltrami_operator_mod, only: beltrami_operator, op_bound_to, op_dimension, op_move, op_release

  implicit none(type, external)

  private
  public :: factor_cache_manager
  public :: factor_cache_acquire, factor_cache_insert, factor_cache_invalidate
  public :: factor_cache_clear, factor_cache_diagnostics

  type :: factor_cache_key
    integer :: volume = 0
    integer :: dimension = 0
    integer :: geometry_generation = 0
    real(rkind) :: mu = 0.0_rkind
  end type factor_cache_key

  type :: factor_cache_entry
    type(beltrami_operator) :: factor
    type(factor_cache_key) :: key
    logical :: occupied = .false.
  end type factor_cache_entry

  !> Independently instantiable manager used by production and lifecycle tests.
  type factor_cache_manager
    private
    type(factor_cache_entry), allocatable :: entries(:)
    integer, allocatable :: geometry_generation(:)
    integer :: hits = 0
    integer :: misses = 0
    integer :: insertions = 0
    integer :: invalidations = 0
  contains
    procedure :: acquire => manager_acquire
    procedure :: insert => manager_insert
    procedure :: invalidate => manager_invalidate
    procedure :: clear => manager_clear
    procedure :: diagnostics => manager_diagnostics
    procedure, private :: drop_slot => manager_drop_slot
    procedure, private :: ensure_capacity => manager_ensure_capacity
    final :: manager_finalize
  end type factor_cache_manager

  type(factor_cache_manager), save :: process_cache

contains

  !> Move a matching cached factor into op. A miss leaves op unchanged.
  logical function manager_acquire(self, enabled, vvol, mu, NN, op) result(hit)
    class(factor_cache_manager), intent(inout) :: self
    logical, intent(in) :: enabled
    integer, intent(in) :: vvol, NN
    real(rkind), intent(in) :: mu
    type(beltrami_operator), intent(inout) :: op

    hit = .false.
    if (.not. enabled) return
    if (.not. allocated(self%entries)) then
      self%misses = self%misses + 1
      return
    end if
    if (vvol < 1 .or. vvol > size(self%entries)) then
      self%misses = self%misses + 1
      return
    end if
    if (.not. self%entries(vvol)%occupied) then
      self%misses = self%misses + 1
      return
    end if
    if (self%entries(vvol)%key%volume /= vvol &
        .or. self%entries(vvol)%key%dimension /= NN &
        .or. self%entries(vvol)%key%mu /= mu &
        .or. self%entries(vvol)%key%geometry_generation &
            /= self%geometry_generation(vvol)) then
      call self%drop_slot(vvol)
      self%misses = self%misses + 1
      return
    end if

    call op_move(self%entries(vvol)%factor, op)
    self%entries(vvol)%occupied = .false.
    self%hits = self%hits + 1
    hit = .true.
  end function manager_acquire

  !> Move op into its volume slot. When disabled or inconsistent, op remains with the caller.
  subroutine manager_insert(self, enabled, nvol, vvol, mu, op)
    class(factor_cache_manager), intent(inout) :: self
    logical, intent(in) :: enabled
    integer, intent(in) :: nvol, vvol
    real(rkind), intent(in) :: mu
    type(beltrami_operator), intent(inout) :: op

    integer :: NN

    if (.not. enabled) return
    if (vvol < 1 .or. vvol > nvol) return
    NN = op_dimension(op)
    if (.not. op_bound_to(op, vvol, NN)) return

    call self%ensure_capacity(nvol)
    call self%drop_slot(vvol)
    call op_move(op, self%entries(vvol)%factor)
    self%entries(vvol)%key%volume = vvol
    self%entries(vvol)%key%dimension = NN
    self%entries(vvol)%key%mu = mu
    self%entries(vvol)%key%geometry_generation = self%geometry_generation(vvol)
    self%entries(vvol)%occupied = .true.
    self%insertions = self%insertions + 1
  end subroutine manager_insert

  !> Invalidate one volume and advance its explicit geometry generation.
  subroutine manager_invalidate(self, vvol)
    class(factor_cache_manager), intent(inout) :: self
    integer, intent(in) :: vvol

    if (.not. allocated(self%entries)) return
    if (vvol < 1 .or. vvol > size(self%entries)) return
    call self%drop_slot(vvol)
    self%geometry_generation(vvol) = self%geometry_generation(vvol) + 1
    self%invalidations = self%invalidations + 1
  end subroutine manager_invalidate

  !> Release every owned factor and reset the manager's capacity.
  subroutine manager_clear(self)
    class(factor_cache_manager), intent(inout) :: self

    integer :: vvol

    if (allocated(self%entries)) then
      do vvol = 1, size(self%entries)
        call self%drop_slot(vvol)
      end do
      deallocate(self%entries, self%geometry_generation)
    end if
  end subroutine manager_clear

  !> Report cache capacity, occupancy, and ownership-operation counters.
  subroutine manager_diagnostics(self, capacity, occupied, hits, misses, insertions, invalidations)
    class(factor_cache_manager), intent(in) :: self
    integer, intent(out) :: capacity, occupied, hits, misses, insertions, invalidations

    if (allocated(self%entries)) then
      capacity = size(self%entries)
      occupied = count(self%entries%occupied)
    else
      capacity = 0
      occupied = 0
    end if
    hits = self%hits
    misses = self%misses
    insertions = self%insertions
    invalidations = self%invalidations
  end subroutine manager_diagnostics

  subroutine manager_drop_slot(self, vvol)
    class(factor_cache_manager), intent(inout) :: self
    integer, intent(in) :: vvol

    call op_release(self%entries(vvol)%factor)
    self%entries(vvol)%key = factor_cache_key()
    self%entries(vvol)%occupied = .false.
  end subroutine manager_drop_slot

  subroutine manager_ensure_capacity(self, nvol)
    class(factor_cache_manager), intent(inout) :: self
    integer, intent(in) :: nvol

    if (allocated(self%entries)) then
      if (size(self%entries) == nvol) return
      call self%clear()
    end if
    allocate(self%entries(nvol), self%geometry_generation(nvol))
    self%geometry_generation = 0
  end subroutine manager_ensure_capacity

  subroutine manager_finalize(self)
    type(factor_cache_manager), intent(inout) :: self

    call self%clear()
  end subroutine manager_finalize

  !> Acquire from the process-owned manager; ownership moves to op on a hit.
  logical function factor_cache_acquire(vvol, mu, NN, op) result(hit)
    integer, intent(in) :: vvol, NN
    real(rkind), intent(in) :: mu
    type(beltrami_operator), intent(inout) :: op

    hit = process_cache%acquire(Lbeltrami_factor_cache, vvol, mu, NN, op)
  end function factor_cache_acquire

  !> Return op to the process-owned manager; ownership moves only when caching is enabled.
  subroutine factor_cache_insert(vvol, mu, op)
    integer, intent(in) :: vvol
    real(rkind), intent(in) :: mu
    type(beltrami_operator), intent(inout) :: op

    call process_cache%insert(Lbeltrami_factor_cache, Mvol, vvol, mu, op)
  end subroutine factor_cache_insert

  !> Drop one process-cache slot and advance its geometry generation.
  subroutine factor_cache_invalidate(vvol)
    integer, intent(in) :: vvol

    call process_cache%invalidate(vvol)
  end subroutine factor_cache_invalidate

  !> Release all process-cache factors in one manager operation.
  subroutine factor_cache_clear()

    call process_cache%clear()
  end subroutine factor_cache_clear

  !> Report diagnostics for the process-owned manager.
  subroutine factor_cache_diagnostics(capacity, occupied, hits, misses, insertions, invalidations)
    integer, intent(out) :: capacity, occupied, hits, misses, insertions, invalidations

    call process_cache%diagnostics(capacity, occupied, hits, misses, insertions, invalidations)
  end subroutine factor_cache_diagnostics

end module beltrami_factor_cache_mod
