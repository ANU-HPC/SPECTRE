module errors_mod
#include "../mpi_precomp.h"
  USE_MPI

  use io_mod, only: stderr

  implicit none

  private
  public :: fatal

contains

  !> Abort run if condition is met
  subroutine fatal(condition, filename, error_message)
    !> Condition triggering the abort
    logical, intent(in) :: condition
    !> Filename where the error was detected
    character(len=*), intent(in) :: filename
    !> Message to be displayed
    character(len=*), intent(in) :: error_message

    integer, parameter :: error_code = 1
    integer :: ierr

    if (condition) then
      write(stderr, '(3A)')filename, " : ", error_message
      call mpi_abort(MPI_COMM_WORLD, error_code, ierr)
    end if
  end subroutine fatal
end module errors_mod
