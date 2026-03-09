module io_mod
  use iso_fortran_env, only: error_unit, input_unit, output_unit

  implicit none

  private
  public :: mute_output
  
  !> Unit to output to standard error
  integer, parameter, public :: stderr = error_unit
  !> Unit to output to standard output
  integer, parameter, public :: stdout = output_unit
  !> Unit for reading from standard input
  integer, parameter, public :: stdin = input_unit

  !> Unit used for all the standard output
  !> By default output is done to stdout, but it can be muted using the \a mute_output() routine.
  integer, public :: ounit = stdout


  integer, parameter, public :: iunit = stdin !< input
  integer, parameter, public :: gunit = 13 !< wall geometry
  integer, parameter, public :: aunit = 11 !< vector potential
  integer, parameter, public :: dunit = 12 !< derivative matrix
  integer, parameter, public :: hunit = 14 !< eigenvalues of Hessian
  integer, parameter, public :: munit = 14 !< matrix elements of Hessian
  integer, parameter, public :: lunit = 20 !< local unit
  integer, parameter, public :: vunit = 15 !< for examination of adaptive quadrature
  integer, parameter :: null_unit = 99 !< Unit used to open /dev/null

contains

  subroutine mute_output(mute)
    !> If True, the Fortran output will be redirected to /dev/null, i.e. tunred off
    logical, intent(in) :: mute

    integer :: ios

    if (mute) then
      ounit = null_unit
      open(ounit, file="/dev/null", status="old", action="write", iostat=ios) 
      if (ios /= 0) then
        write(stderr, *) "Unable to open /dev/null for muting output!"
      end if
    else
      close(ounit)
      ounit = stdout
    end if
  end subroutine mute_output
end module io_mod
