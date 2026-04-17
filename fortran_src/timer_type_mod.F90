module timer_type_mod
#include "mpi_precomp.h"
  USE_MPI

  use io_mod, only: stderr
  use precisions_mod, only: rkind

  implicit none

  private
  public :: TTimer

  type TTimer
    private
    real(rkind) :: t0
    real(rkind) :: t1
    real(rkind) :: total_time
    logical :: is_on

    contains

    procedure, public :: get_total_time
    procedure, public :: get_current_time
    procedure, public :: timer_on
    procedure, public :: timer_off
    procedure, public :: reset
  end type TTimer

  interface TTimer
    procedure :: constructor
  end interface TTimer

contains

  type(TTimer) pure function constructor()
    constructor%total_time = 0.0_rkind
    constructor%is_on = .false.
  end function constructor

  real(rkind) pure function get_total_time(this)
    class(TTimer), intent(in) :: this

    get_total_time = this%total_time
  end function get_total_time

  real(rkind) function get_current_time(this)
    class(TTimer), intent(in) :: this

    get_current_time = -1.0_rkind
    if (this%is_on) then

      get_current_time = this%total_time + MPI_Wtime() - this%t0
    else
      write(stderr, *)"Timer needs to be activated to get current time!"
    end if
  end function get_current_time

  subroutine timer_on(this)
    class(TTimer), intent(inout) :: this

    this%is_on = .true.
    this%t0 = MPI_Wtime()
  end subroutine timer_on

  subroutine timer_off(this)
    class(TTimer), intent(inout) :: this

    real(rkind) :: total_tmp
    integer :: ierr

    if (this%is_on) then
      this%t1 = MPI_Wtime()
      this%is_on = .false.

      this%total_time = this%total_time + this%t1 - this%t0
    else
      write(stderr, *)"Timer needs to be activated before deactivating it!"
    end if
  end subroutine timer_off

  subroutine reset(this)
    class(TTimer), intent(inout) :: this

    this%total_time = 0.0_rkind
  end subroutine reset
end module timer_type_mod
