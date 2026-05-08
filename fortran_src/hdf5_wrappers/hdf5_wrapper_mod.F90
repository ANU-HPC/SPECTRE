! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

!> Provide wrapper for HDF5
module hdf5_wrapper_mod
  use hdf5

  use precisions_mod, only: rkind
  use conversion_mod, only: num2str
  use errors_mod, only: fatal

  implicit none

  private
  public :: hdf5_close_group, hdf5_define_group, hdf5_add_object_description, hdf5_add_dataset_description, &
      hdf5_write_logical, hdf5_write_logical_leave_open, hdf5_write_integer, hdf5_write_integer_leave_open, hdf5_write_real, &
      hdf5_write_real_leave_open, hdf5_write_real_2D, hdf5_write_real_2D_leave_open, hdf5_write_real_3D

contains

  subroutine hdf5_define_group(parent_group_id, group_name, group_id)
    integer(hid_t), intent(in) :: parent_group_id
    character(len=*), intent(in) :: group_name
    integer(hid_t), intent(out) :: group_id

    logical :: group_exists
    integer :: hdf5_ierr

    call h5lexists_f(parent_group_id, group_name, group_exists, hdf5_ierr)

    if (.not.group_exists) then
      ! if group does not exist, create it
      call h5gcreate_f(parent_group_id, group_name, group_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5create_f from hdf5_define_group")
    else
      ! if the group already exists, open it
      call h5gopen_f(parent_group_id, group_name, group_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5open_f from hdf5_define_group")
    end if
  end subroutine hdf5_define_group

  subroutine hdf5_close_group(group_id)
    integer(hid_t), intent(in) :: group_id

    integer :: hdf5_ierr

    call h5gclose_f(group_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5gclose_f from hdf5_close_group")
  end subroutine hdf5_close_group

  subroutine hdf5_add_object_description(group_id, description)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: description

    integer :: hdf5_ierr
    integer(hsize_t) :: description_length
    integer, parameter :: att_rank = 1
    integer(hsize_t), dimension(att_rank) :: att_dims = [1]
    integer(hid_t) :: att_id
    integer(hid_t) :: att_space_id
    integer(hid_t) :: att_type_id
    character(LEN=15), parameter :: att_name = "description"

    description_length = len(description)

    call h5screate_simple_f(att_rank, att_dims, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_add_object_description")

    call h5tcopy_f(H5T_NATIVE_CHARACTER, att_type_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tcopy_f from hdf5_add_object_description")

    call h5tset_size_f(att_type_id, description_length, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tset_size_f from hdf5_add_object_description")

    call h5acreate_f(group_id, att_name, att_type_id, att_space_id, att_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5acreate_f from hdf5_add_object_description")

    call h5awrite_f(att_id, att_type_id, description, att_dims, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5awrite_f from hdf5_add_object_description")

    call h5aclose_f(att_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5aclose_f from hdf5_add_object_description")

    call h5tclose_f(att_type_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tclose_f from hdf5_add_object_description")

    call h5sclose_f(att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5sclose_f from hdf5_add_object_description")
  end subroutine hdf5_add_object_description

  subroutine hdf5_add_dataset_description(dataset_id, description)
    integer(hid_t), intent(in) :: dataset_id
    character(len=*), intent(in) :: description

    integer :: hdf5_ierr
    integer(hsize_t) :: description_length
    integer, parameter :: att_rank = 1
    integer(hsize_t), dimension(att_rank) :: att_dims = [1]
    integer(hid_t) :: att_id
    integer(hid_t) :: att_space_id
    integer(hid_t) :: att_type_id
    character(LEN=15), parameter :: att_name = "description"

    description_length = len(description)

    call h5screate_simple_f(att_rank, att_dims, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_add_dataset_description")

    call h5tcopy_f(H5T_NATIVE_CHARACTER, att_type_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tcopy_f from hdf5_add_dataset_description")

    call h5tset_size_f(att_type_id, description_length, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tset_size_f from hdf5_add_dataset_description")

    call h5acreate_f(dataset_id, att_name, att_type_id, att_space_id, att_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5acreate_f from hdf5_add_dataset_description")

    call h5awrite_f(att_id, att_type_id, description, att_dims, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5awrite_f from hdf5_add_dataset_description")

    call h5aclose_f(att_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5aclose_f from hdf5_add_dataset_description")

    call h5tclose_f(att_type_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5tclose_f from hdf5_add_dataset_description")

    call h5sclose_f(att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5sclose_f from hdf5_add_dataset_description")

    call h5dclose_f(dataset_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_add_dataset_description")
  end subroutine hdf5_add_dataset_description

  subroutine hdf5_write_logical(group_id, dataset, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    logical, intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    integer(hid_t) :: dataset_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_logical")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_logical")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_logical")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_INTEGER, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_logical")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, merge(1, 0, value), value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_logical")

    call h5dclose_f(dataset_id, hdf5_ierr)! terminate dataset
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_write_logical")
  end subroutine hdf5_write_logical

  subroutine hdf5_write_logical_leave_open(group_id, dataset, dataset_id, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    integer(hid_t), intent(out) :: dataset_id
    logical, intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_logical")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_logical")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_logical")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_INTEGER, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_logical")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, merge(1, 0, value), value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_logical")
  end subroutine hdf5_write_logical_leave_open

  subroutine hdf5_write_integer(group_id, dataset, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    integer, intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    integer(hid_t) :: dataset_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_integer")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_integer")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_integer")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_INTEGER, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_integer")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_integer")

    call h5dclose_f(dataset_id, hdf5_ierr)! terminate dataset
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_write_integer")
  end subroutine hdf5_write_integer

  subroutine hdf5_write_integer_leave_open(group_id, dataset, dataset_id, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    integer(hid_t), intent(out) :: dataset_id
    integer, intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_integer")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_integer")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_integer")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_INTEGER, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_integer")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_integer")
  end subroutine hdf5_write_integer_leave_open

  subroutine hdf5_write_real(group_id, dataset, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    real(rkind), intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    integer(hid_t) :: dataset_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_real")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_DOUBLE, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_real")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_real")

    call h5dclose_f(dataset_id, hdf5_ierr)! terminate dataset
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_write_real")
  end subroutine hdf5_write_real

  subroutine hdf5_write_real_leave_open(group_id, dataset, dataset_id, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    integer(hid_t), intent(out) :: dataset_id
    real(rkind), intent(in) :: value(:)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    logical :: var_exists
    integer, parameter :: rank = 1
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = [int(size(value), hsize_t)]
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_real")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_DOUBLE, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_real")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_real")
  end subroutine hdf5_write_real_leave_open

  subroutine hdf5_write_real_2D(group_id, dataset, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    real(rkind), intent(in) :: value(:, :)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    integer(hid_t) :: dataset_id
    logical :: var_exists
    integer, parameter :: rank = 2
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = int(shape(value), hsize_t)
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_real")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_DOUBLE, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_real")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_real")

    call h5dclose_f(dataset_id, hdf5_ierr)! terminate dataset
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_write_real")
  end subroutine hdf5_write_real_2D

  subroutine hdf5_write_real_2D_leave_open(group_id, dataset, dataset_id, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    integer(hid_t), intent(out) :: dataset_id
    real(rkind), intent(in) :: value(:, :)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    logical :: var_exists
    integer, parameter :: rank = 2
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = int(shape(value), hsize_t)
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_real")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_DOUBLE, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_real")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_real")
  end subroutine hdf5_write_real_2D_leave_open

  subroutine hdf5_write_real_3D(group_id, dataset, value)
    integer(hid_t), intent(in) :: group_id
    character(len=*), intent(in) :: dataset
    real(rkind), intent(in) :: value(:, :, :)

    integer :: hdf5_ierr
    integer(hid_t) :: att_space_id
    integer(hid_t) :: dataset_id
    logical :: var_exists
    integer, parameter :: rank = 3
    integer, parameter :: internalHdf5Msg = 0
    integer(hsize_t) :: value_size(rank)

    value_size = int(shape(value), hsize_t)
    call h5screate_simple_f(rank, value_size, att_space_id, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5screate_simple_f from hdf5_write_real")

    ! temporarily disable error printing to not confuse users
    call h5eset_auto_f(0, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! check if dataset can be opened
    var_exists = .false.
    call h5dopen_f(group_id, dataset, dataset_id, hdf5_ierr)
    if (hdf5_ierr >= 0) then
      var_exists = .true.
    end if

    ! re-establish previous state of error printing to be sensitive to "real" errors
    call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5eset_auto_f from hdf5_write_real")

    ! if the dataset does not exist already, create it. Otherwise, it should be open already
    if (.not.var_exists) then
      call h5dcreate_f(group_id, dataset, H5T_NATIVE_DOUBLE, att_space_id, dataset_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dcreate_f from hdf5_write_real")
    end if

    call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, value, value_size, hdf5_ierr)
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dwrite_f from hdf5_write_real")

    call h5dclose_f(dataset_id, hdf5_ierr)! terminate dataset
    call fatal(hdf5_ierr /= 0, __FILE__, "Error calling h5dclose_f from hdf5_write_real")
  end subroutine hdf5_write_real_3D
end module hdf5_wrapper_mod
