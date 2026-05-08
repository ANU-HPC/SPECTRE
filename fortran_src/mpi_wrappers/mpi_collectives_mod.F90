! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module mpi_collectives_mod
#include "../mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind

  implicit none

  private
  public :: init_mpi_collective_mod, broadcast

  interface broadcast
    module procedure bcast_logical_0D
    module procedure bcast_logical_1D

    module procedure bcast_integer_0D
    module procedure bcast_integer_1D

    module procedure bcast_real_0D
    module procedure bcast_real_1D
    module procedure bcast_real_2D
    module procedure bcast_real_3D
    module procedure bcast_real_4D

    module procedure bcast_character_1D
  end interface broadcast

  integer :: ierr
  MPI_COMM_TYPE :: communicator = MPI_COMM_NULL

contains

  !> Initialize the module
  subroutine init_mpi_collective_mod(comm)
    MPI_COMM_TYPE, intent(in) :: comm

    communicator = comm
  end subroutine init_mpi_collective_mod

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! LOGICAL                                                                  !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Broadcast a 0D logical
  subroutine bcast_logical_0D(val, source)
    !> Value to send
    logical, intent(in) :: val
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, 1, MPI_LOGICAL, source, communicator, ierr)
  end subroutine bcast_logical_0D

  !> Broadcast a 1D logical
  subroutine bcast_logical_1D(val, count, source)
    !> Value to send
    logical, intent(in) :: val(:)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_LOGICAL, source, communicator, ierr)
  end subroutine bcast_logical_1D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! INTEGER                                                                  !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Broadcast a 0D integer
  subroutine bcast_integer_0D(val, source)
    !> Value to send
    integer, intent(in) :: val
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, 1, MPI_INTEGER, source, communicator, ierr)
  end subroutine bcast_integer_0D

  !> Broadcast a 1D integer
  subroutine bcast_integer_1D(val, count, source)
    !> Value to send
    integer, intent(in) :: val(:)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_INTEGER, source, communicator, ierr)
  end subroutine bcast_integer_1D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! REAL                                                                     !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Broadcast a 0D real
  subroutine bcast_real_0D(val, source)
    !> Value to send
    real(rkind), intent(in) :: val
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, 1, MPI_DOUBLE_PRECISION, source, communicator, ierr)
  end subroutine bcast_real_0D

  !> Broadcast a 1D real
  subroutine bcast_real_1D(val, count, source)
    !> Value to send
    real(rkind), intent(in) :: val(:)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_DOUBLE_PRECISION, source, communicator, ierr)
  end subroutine bcast_real_1D

  !> Broadcast a 2D real
  subroutine bcast_real_2D(val, count, source)
    !> Value to send
    real(rkind), intent(in) :: val(:, :)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_DOUBLE_PRECISION, source, communicator, ierr)
  end subroutine bcast_real_2D

  !> Broadcast a 3D real
  subroutine bcast_real_3D(val, count, source)
    !> Value to send
    real(rkind), intent(in) :: val(:, :, :)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_DOUBLE_PRECISION, source, communicator, ierr)
  end subroutine bcast_real_3D

  !> Broadcast a 4D real
  subroutine bcast_real_4D(val, count, source)
    !> Value to send
    real(rkind), intent(in) :: val(:, :, :, :)
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_DOUBLE_PRECISION, source, communicator, ierr)
  end subroutine bcast_real_4D

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! CHARACTER                                                                     !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Broadcast a 1D character
  subroutine bcast_character_1D(val, count, source)
    !> Value to send
    character(len=*), intent(in) :: val
    !> Number of values to send
    integer, intent(in) :: count
    !> MPI rank being the source of the broadcast
    integer, intent(in) :: source

    call mpi_bcast(val, count, MPI_CHARACTER, source, communicator, ierr)
  end subroutine bcast_character_1D
end module mpi_collectives_mod
