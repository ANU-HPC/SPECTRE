! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module conversion_mod

  implicit none

  private
  public :: num2str

contains
  !> Convert an integer to a string
  !
  !> Note that the max length is 20 digits.
  function num2str(n)result(str)
    implicit none
    integer, intent(in) :: n
    character(len=20) :: str

    write(str, '(I20)')n
    str = adjustl(str)
  end function num2str
end module conversion_mod
