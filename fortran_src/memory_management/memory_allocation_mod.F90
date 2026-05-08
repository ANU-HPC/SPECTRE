! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module memory_allocation_mod
  use precisions_mod, only: rkind
  use typedefns_mod, only: subgrid

  implicit none

  private
  public :: nallocate, sallocate

  interface nallocate
    module procedure nallocate_real_1D
    module procedure nallocate_real_2D
    module procedure nallocate_real_3D
    module procedure nallocate_real_4D
    module procedure nallocate_real_5D

    module procedure nallocate_integer_1D
    module procedure nallocate_integer_2D
    module procedure nallocate_integer_3D

    module procedure nallocate_logical_1D

    module procedure nallocate_complex_2D

    module procedure nallocate_subgrid_3D
  end interface nallocate

  interface sallocate
    module procedure sallocate_real_1D
    module procedure sallocate_real_2D
    module procedure sallocate_real_3D
    module procedure sallocate_real_4D
    module procedure sallocate_real_5D

    module procedure sallocate_integer_1D
    module procedure sallocate_integer_2D
    module procedure sallocate_integer_3D

    module procedure sallocate_logical_1D

    module procedure sallocate_complex_2D
  end interface sallocate

contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! NALLOCATE REAL                                                           !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D real array
  pure subroutine nallocate_real_1D(array, sind1, eind1)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1

    allocate(array(sind1:eind1))
  end subroutine nallocate_real_1D

  !> Allocate 2D real array
  pure subroutine nallocate_real_2D(array, sind1, eind1, sind2, eind2)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2

    allocate(array(sind1:eind1, sind2:eind2))
  end subroutine nallocate_real_2D

  !> Allocate 3D real array
  pure subroutine nallocate_real_3D(array, sind1, eind1, sind2, eind2, sind3, eind3)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3

    allocate(array(sind1:eind1, sind2:eind2, sind3:eind3))
  end subroutine nallocate_real_3D

  !> Allocate 4D real array
  pure subroutine nallocate_real_4D(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4

    allocate(array(sind1:eind1, sind2:eind2, sind3:eind3, sind4:eind4))
  end subroutine nallocate_real_4D

  !> Allocate 5D real array
  pure subroutine nallocate_real_5D(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5

    allocate(array(sind1:eind1, sind2:eind2, sind3:eind3, sind4:eind4, sind5:eind5))
  end subroutine nallocate_real_5D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! NALLOCATE INTEGR                                                         !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D integer array
  pure subroutine nallocate_integer_1D(array, sind1, eind1)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1

    allocate(array(sind1:eind1))
  end subroutine nallocate_integer_1D

  !> Allocate 2D integer array
  pure subroutine nallocate_integer_2D(array, sind1, eind1, sind2, eind2)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2

    allocate(array(sind1:eind1, sind2:eind2))
  end subroutine nallocate_integer_2D

  !> Allocate 3D integer array
  pure subroutine nallocate_integer_3D(array, sind1, eind1, sind2, eind2, sind3, eind3)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3

    allocate(array(sind1:eind1, sind2:eind2, sind3:eind3))
  end subroutine nallocate_integer_3D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! NALLOCATE LOGICAL                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D logical array
  pure subroutine nallocate_logical_1D(array, sind1, eind1)
    !> Array to allocate
    logical, allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1

    allocate(array(sind1:eind1))
  end subroutine nallocate_logical_1D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! NALLOCATE COMPLEX                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 2D complex array
  pure subroutine nallocate_complex_2D(array, sind1, eind1, sind2, eind2)
    !> Array to allocate
    complex(rkind), allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2

    allocate(array(sind1:eind1, sind2:eind2))
  end subroutine nallocate_complex_2D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! NALLOCATE SUBGRID                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 3D subgrid array
  pure subroutine nallocate_subgrid_3D(array, sind1, eind1, sind2, eind2, sind3, eind3)
    !> Array to allocate
    type(subgrid), allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3

    allocate(array(sind1:eind1, sind2:eind2, sind3:eind3))
  end subroutine nallocate_subgrid_3D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! SALLOCATE REAL                                                           !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D real array and set it to a value
  pure subroutine sallocate_real_1D(array, sind1, eind1, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1
    !> Value to set the array to
    real(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1)
    array = val
  end subroutine sallocate_real_1D

  !> Allocate 2D real array and set it to a value
  pure subroutine sallocate_real_2D(array, sind1, eind1, sind2, eind2, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2
    !> Value to set the array to
    real(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2)
    array = val
  end subroutine sallocate_real_2D

  !> Allocate 3D real array and set it to a value
  pure subroutine sallocate_real_3D(array, sind1, eind1, sind2, eind2, sind3, eind3, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3
    !> Value to set the array to
    real(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3)
    array = val
  end subroutine sallocate_real_3D

  !> Allocate 4D real array and set it to a value
  pure subroutine sallocate_real_4D(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4
    !> Value to set the array to
    real(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4)
    array = val
  end subroutine sallocate_real_4D

  !> Allocate 5D real array and set it to a value
  pure subroutine sallocate_real_5D(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5
    !> Value to set the array to
    real(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5)
    array = val
  end subroutine sallocate_real_5D

  !> Allocate 1D real array and set it to a valued array
  pure subroutine sallocate_real_1D_array(array, sind1, eind1, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1
    !> Value to set the array to
    real(rkind), intent(in) :: val(:)

    call nallocate(array, sind1, eind1)
    array(:) = val(:)
  end subroutine sallocate_real_1D_array

  !> Allocate 2D real array and set it to a valued array
  pure subroutine sallocate_real_2D_array(array, sind1, eind1, sind2, eind2, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2
    !> Value to set the array to
    real(rkind), intent(in) :: val(:, :)

    call nallocate(array, sind1, eind1, sind2, eind2)
    array(:, :) = val(:, :)
  end subroutine sallocate_real_2D_array

  !> Allocate 3D real array and set it to a valued array
  pure subroutine sallocate_real_3D_array(array, sind1, eind1, sind2, eind2, sind3, eind3, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3
    !> Value to set the array to
    real(rkind), intent(in) :: val(:, :, :)

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3)
    array(:, :, :) = val(:, :, :)
  end subroutine sallocate_real_3D_array

  !> Allocate 4D real array and set it to a valued array
  pure subroutine sallocate_real_4D_array(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4
    !> Value to set the array to
    real(rkind), intent(in) :: val(:, :, :, :)

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4)
    array(:, :, :, :) = val(:, :, :, :)
  end subroutine sallocate_real_4D_array

  !> Allocate 5D real array and set it to a valued array
  pure subroutine sallocate_real_5D_array(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5, val)
    !> Array to allocate
    real(rkind), allocatable, intent(out) :: array(:, :, :, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5
    !> Value to set the array to
    real(rkind), intent(in) :: val(:, :, :, :, :)

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3, sind4, eind4, sind5, eind5)
    array(:, :, :, :, :) = val(:, :, :, :, :)
  end subroutine sallocate_real_5D_array

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! SALLOCATE INTEGER                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D real array and set it to a value
  pure subroutine sallocate_integer_1D(array, sind1, eind1, val)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1
    !> Value to set the array to
    integer, intent(in) :: val

    call nallocate(array, sind1, eind1)
    array = val
  end subroutine sallocate_integer_1D

  !> Allocate 2D integer array and set it to a value
  pure subroutine sallocate_integer_2D(array, sind1, eind1, sind2, eind2, val)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2
    !> Value to set the array to
    integer, intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2)
    array = val
  end subroutine sallocate_integer_2D

  !> Allocate 3D integer array and set it to a value
  pure subroutine sallocate_integer_3D(array, sind1, eind1, sind2, eind2, sind3, eind3, val)
    !> Array to allocate
    integer, allocatable, intent(out) :: array(:, :, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2, sind3, eind3
    !> Value to set the array to
    integer, intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2, sind3, eind3)
    array = val
  end subroutine sallocate_integer_3D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! SALLOCATE LOGICAL                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 1D real array and set it to a value
  pure subroutine sallocate_logical_1D(array, sind1, eind1, val)
    !> Array to allocate
    logical, allocatable, intent(out) :: array(:)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1
    !> Value to set the array to
    logical, intent(in) :: val

    call nallocate(array, sind1, eind1)
    array = val
  end subroutine sallocate_logical_1D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! SALLOCATE COMPLEX                                                        !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Allocate 2D complex array and set it to a value
  pure subroutine sallocate_complex_2D(array, sind1, eind1, sind2, eind2, val)
    !> Array to allocate
    complex(rkind), allocatable, intent(out) :: array(:, :)
    !> Bounds of the array
    integer, intent(in) :: sind1, eind1, sind2, eind2
    !> Value to set the array to
    complex(rkind), intent(in) :: val

    call nallocate(array, sind1, eind1, sind2, eind2)
    array = val
  end subroutine sallocate_complex_2D
end module memory_allocation_mod
