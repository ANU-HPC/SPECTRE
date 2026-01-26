!> \brief type definitions for custom datatypes
!> \ingroup grp_global
module typedefns_mod
  use precisions_mod, only: rkind

  private
  public :: subgrid, MatrixLU, derivative

  !> \brief used for quantities which have different resolutions in different volumes, e.g. the vector potential
  type subgrid
    real(rkind), allocatable :: s(:)!< coefficients
    integer, allocatable :: i(:)!< indices
  end type subgrid

  type MatrixLU
    real(rkind), allocatable :: mat(:, :)
    integer, allocatable :: ipivot(:)
  end type MatrixLU

  !> \brief \f${\rm d}\mathbf{B}/{\rm d}\mathbf{X}\f$ (?)
  type derivative
    logical :: L !< what is this?
    integer :: vol !< Used in compute_coordinates(): required for global constraint force gradient evaluation
    integer :: innout !< what is this?
    integer :: ii !< what is this?
    integer :: irz !< what is this?
    integer :: issym !< what is this?
  end type derivative

end module typedefns_mod
