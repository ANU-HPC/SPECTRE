! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module sphdf5_mod
  use hdf5

  use precisions_mod, only: rkind
  use input_list_mod, only: Wsphdf5, Wmacros
  use io_mod, only: ounit, stderr
  use cputiming_mod, only: Tsphdf5
  use allglobal_mod, only: myid, cpus, MPI_COMM_SPEC, filename_h5, skip_write, im, in, ims, ins, iRbc, iRbs, iZbc, iZbs,&
      & beltramierror, TT, Btemn, Bzemn, Btomn, Bzomn, mns, ForceErr, BnsErr, Rij, Zij, sg, inv_jac, guvij
  use constants_mod, only: version
  use memory_allocation_mod, only: sallocate
  use errors_mod, only: fatal
  use hdf5_wrapper_mod, only: hdf5_close_group, hdf5_define_group, hdf5_add_object_description,&
      & hdf5_add_dataset_description, hdf5_write_logical, hdf5_write_integer, hdf5_write_integer_leave_open,&
      & hdf5_write_real, hdf5_write_real_leave_open, hdf5_write_real_2D, hdf5_write_real_2D_leave_open,&
      & hdf5_write_real_3D, hdf5_write_logical_leave_open
  use volume_mod, only: lregion
  use conversion_mod, only: num2str
  use magnetic_field_mod, only: compute_magnetic_field
  use geometry_mod, only: compute_coordinates

  implicit none

  logical, parameter :: hdfDebug = .false. !< global flag to enable verbal diarrhea commenting HDF5 operations
  integer, parameter :: internalHdf5Msg = 0 !< 1: print internal HDF5 error messages: 0: only error messages from sphdf5

  integer :: hdf5_ierr !< error flag for HDF5 library
  integer :: rank !< rank of data to write using macros
  integer(hid_t) :: file_id !< default file ID used in macros
  integer(hid_t) :: space_id !< default dataspace ID used in macros
  integer(hid_t) :: dset_id !< default dataset ID used in macros
  integer(hsize_t) :: onedims(1:1)!< dimension specifier for zero-dimensional data used in macros
  integer(hsize_t) :: twodims(1:2)!< dimension specifier for two-dimensional data used in macros
  integer(hsize_t) :: threedims(1:3)!< dimension specifier for three-dimensional data used in macros
  logical :: grp_exists !< flags used to signal if a group already exists
  logical :: var_exists !< flags used to signal if a variable already exists


  integer, parameter :: rankP = 3 !< rank of Poincare data
  integer, parameter :: rankT = 2 !< rank of rotational transform data

  integer(hid_t) :: grpPoincare !< group for Poincare data
  integer(HID_T) :: dset_id_t !< Dataset identifier for \f$\theta\f$ coordinate of field line following
  integer(HID_T) :: dset_id_s !< Dataset identifier for \f$s\f$ coordinate of field line following
  integer(HID_T) :: dset_id_R !< Dataset identifier for \f$R\f$ coordinate of field line following
  integer(HID_T) :: dset_id_Z !< Dataset identifier for \f$Z\f$ coordinate of field line following
  integer(HID_T) :: dset_id_success !< Dataset identifier for success flag of trajectories to follow
  integer(HID_T) :: filespace_t !< Dataspace identifier in file for \f$\theta\f$ coordinate of field line following
  integer(HID_T) :: filespace_s !< Dataspace identifier in file for \f$s\f$ coordinate of field line following
  integer(HID_T) :: filespace_R !< Dataspace identifier in file for \f$R\f$ coordinate of field line following
  integer(HID_T) :: filespace_Z !< Dataspace identifier in file for \f$Z\f$ coordinate of field line following
  integer(HID_T) :: filespace_success !< Dataspace identifier in file for success flag of trajectories to follow
  integer(HID_T) :: memspace_t !< Dataspace identifier in memory for \f$\theta\f$ coordinate of field line following
  integer(HID_T) :: memspace_s !< Dataspace identifier in memory for \f$s\f$ coordinate of field line following
  integer(HID_T) :: memspace_R !< Dataspace identifier in memory for \f$R\f$ coordinate of field line following
  integer(HID_T) :: memspace_Z !< Dataspace identifier in memory for \f$Z\f$ coordinate of field line following
  integer(HID_T) :: memspace_success !< Dataspace identifier in memory for success flag of trajectories to follow

  integer(hid_t) :: grpTransform !< group for rotational transform data
  integer(HID_T) :: dset_id_diotadxup !< Dataset identifier for diotadxup (derivative of rotational transform ?)
  integer(HID_T) :: dset_id_fiota !< Dataset identifier for fiota     (              rotational transform ?)
  integer(HID_T) :: filespace_diotadxup !< Dataspace identifier in file for diotadxup
  integer(HID_T) :: filespace_fiota !< Dataspace identifier in file for fiota
  integer(HID_T) :: memspace_diotadxup !< Dataspace identifier in memory for diotadxup
  integer(HID_T) :: memspace_fiota !< Dataspace identifier in memory for fiota

  character(LEN=15), parameter :: aname = "description" !< Attribute name for descriptive info

  integer(HID_T) :: attr_id !< Attribute identifier
  integer(HID_T) :: aspace_id !< Attribute Dataspace identifier
  integer(HID_T) :: atype_id !< Attribute Datatype identifier

  integer, parameter :: arank = 1 !< Attribure rank
  integer(HSIZE_T), dimension(arank) :: adims = [1] !< Attribute dimension

  integer(SIZE_T) :: attrlen !< Length of the attribute string
  character(len=:), allocatable :: attr_data !< Attribute data

contains

  subroutine init_outfile
    if (myid == 0 .and. .not.skip_write) then

      ! initialize Fortran interface to the HDF5 library
      call h5open_f(hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! (en/dis)able HDF5 internal error messages: sphdf5 has its own error messages coming from the macros
      call h5eset_auto_f(internalHdf5Msg, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Create the file
      call h5fcreate_f(trim(filename_h5), H5F_ACC_TRUNC_F, file_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! write version number
      call hdf5_write_real_leave_open(file_id, "version", dset_id, [version])
      call hdf5_add_dataset_description(dset_id, "version of SPECTRE")

    end if

  end subroutine init_outfile

  subroutine mirror_input_to_outfile

    use input_list_mod
    use allglobal_mod, only: Mvol

    integer(hid_t) :: grpInput
    integer(hid_t) :: grpInputPhysics, grpInputNumerics, grpInputLocal, grpInputGlobal, grpInputDiagnostics

    if (myid == 0 .and. .not.skip_write) then

      call hdf5_define_group(file_id, "input", grpInput)
      call hdf5_add_object_description(grpInput, "group for mirrored input data")

      ! the following variables constitute the namelist/physicslist/: note that all variables in namelist need to be
      ! broadcasted in readin
      ! they go into ext.h5/input/physics

      ! the calls used here work as follows:
      ! step 1. HWRITEIV_LO e.g. write(s an) i(nteger) v(ariable) and l(eaves) o(pen) the dataset, so that in
      ! step 2a. an attribute with descr(iptive) information can be attached to the dataset and finally, in
      ! step 2b. the attribute is closed and also we c(lose the) d(ata)set.

      call hdf5_define_group(grpInput, "physics", grpInputPhysics)
      call hdf5_add_object_description(grpInputPhysics, "physics inputs")

      call hdf5_write_integer_leave_open(grpInputPhysics, "Igeometry", dset_id, [Igeometry])
      call hdf5_add_dataset_description(dset_id, "geometry identifier")
      call hdf5_write_logical_leave_open(grpInputPhysics, "enforce_stell_sym", dset_id, [enforce_stell_sym])
      call hdf5_add_dataset_description(dset_id, "stellarator symmetry flag")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Lfreebound", dset_id, [Lfreebound])
      call hdf5_add_dataset_description(dset_id, "free boundary flag")
      call hdf5_write_integer_leave_open(grpInputPhysics, "lvcvacuum", dset_id, [lvcvacuum])
      call hdf5_add_dataset_description(dset_id, "free-boundary vacuum formulation: 0 Picard, 1 direct Laplace-virtual-casing")
      call hdf5_write_real_leave_open(grpInputPhysics, "phiedge", dset_id, [phiedge])
      call hdf5_add_dataset_description(dset_id, "total enclosed toroidal magnetic flux")
      call hdf5_write_real_leave_open(grpInputPhysics, "curtor", dset_id, [curtor])
      call hdf5_add_dataset_description(dset_id, "total enclosed toroidal current")
      call hdf5_write_real_leave_open(grpInputPhysics, "curpol", dset_id, [curpol])
      call hdf5_add_dataset_description(dset_id, "total enclosed poloidal current")
      call hdf5_write_real_leave_open(grpInputPhysics, "gamma", dset_id, [gamma])
      call hdf5_add_dataset_description(dset_id, "adiabatic index")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Nfp", dset_id, [Nfp])
      call hdf5_add_dataset_description(dset_id, "number of stellarator field periods")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Nvol", dset_id, [Nvol])
      call hdf5_add_dataset_description(dset_id, "number of volumes")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Mpol", dset_id, [Mpol])
      call hdf5_add_dataset_description(dset_id, "maximum poloidal mode number")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Ntor", dset_id, [Ntor])
      call hdf5_add_dataset_description(dset_id, "maximum toroidal mode number")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Lrad", dset_id, Lrad(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "degree of radial Chebychev polynomials")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Lconstraint", dset_id, [Lconstraint])
      call hdf5_add_dataset_description(dset_id, "type of constraint to enforce")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Lreflect", dset_id, [Lreflect])
      call hdf5_add_dataset_description(dset_id, "whether to reflect the perturbation on both boundaries for slab      &
          &     geometry")
      call hdf5_write_real_leave_open(grpInputPhysics, "tflux", dset_id, tflux(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "toroidal magnetic flux in volumes")
      call hdf5_write_real_leave_open(grpInputPhysics, "pflux", dset_id, pflux(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "poloidal magnetic flux in volumes")
      call hdf5_write_real_leave_open(grpInputPhysics, "helicity", dset_id, helicity(1:Nvol))
      call hdf5_add_dataset_description(dset_id, "helicity profile")
      call hdf5_write_real_leave_open(grpInputPhysics, "pscale", dset_id, [pscale])
      call hdf5_add_dataset_description(dset_id, "scaling factor for pressure")
      call hdf5_write_real_leave_open(grpInputPhysics, "pressure", dset_id, pressure(1:Nvol))
      call hdf5_add_dataset_description(dset_id, "pressure profile")
      call hdf5_write_integer_leave_open(grpInputPhysics, "Ladiabatic", dset_id, [Ladiabatic])
      call hdf5_add_dataset_description(dset_id, "adiabatic flag")
      call hdf5_write_real_leave_open(grpInputPhysics, "adiabatic", dset_id, adiabatic(1:Nvol))
      call hdf5_add_dataset_description(dset_id, "adiabatic profile (?)")
      call hdf5_write_real_leave_open(grpInputPhysics, "mu", dset_id, mu(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "Beltrami parameter, parallel current profile")
      call hdf5_write_real_leave_open(grpInputPhysics, "Ivolume", dset_id, Ivolume(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "Volume current, externally driven, parallel current profile")
      call hdf5_write_real_leave_open(grpInputPhysics, "Isurf", dset_id, Isurf(1:Mvol))
      call hdf5_add_dataset_description(dset_id, "Surface current, currents that are not volume currents (pressure     &
          &      driven, shielding currents)")
      call hdf5_write_integer_leave_open(grpInputPhysics, "pl", dset_id, pl(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "pl ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "ql", dset_id, ql(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "ql ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "pr", dset_id, pr(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "pr ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "qr", dset_id, qr(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "qr ?")
      call hdf5_write_real_leave_open(grpInputPhysics, "iota", dset_id, iota(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "rotational transform profile on inside of ideal interfaces")
      call hdf5_write_integer_leave_open(grpInputPhysics, "lp", dset_id, lp(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "lp ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "lq", dset_id, lq(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "lq ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "rp", dset_id, rp(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "rp ?")
      call hdf5_write_integer_leave_open(grpInputPhysics, "rq", dset_id, rq(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "rq ?")
      call hdf5_write_real_leave_open(grpInputPhysics, "oita", dset_id, oita(0:Nvol))
      call hdf5_add_dataset_description(dset_id, "rotational transform profile on outside of ideal interfaces")
      call hdf5_write_real_leave_open(grpInputPhysics, "rtor", dset_id, [rtor])
      call hdf5_add_dataset_description(dset_id, "for aspect ratio in slab")
      call hdf5_write_real_leave_open(grpInputPhysics, "rpol", dset_id, [rpol])
      call hdf5_add_dataset_description(dset_id, "for aspect ratio in slab")

      call hdf5_write_real_leave_open(grpInputPhysics, "Rac", dset_id, Rac(0:Ntor))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric coordinate axis R cosine Fourier coefficients")
      call hdf5_write_real_leave_open(grpInputPhysics, "Zas", dset_id, Zas(0:Ntor))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric coordinate axis Z   sine Fourier coefficients")
      call hdf5_write_real_leave_open(grpInputPhysics, "Ras", dset_id, Ras(0:Ntor))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric coordinate axis R   sine Fourier          &
          & coefficients")
      call hdf5_write_real_leave_open(grpInputPhysics, "Zac", dset_id, Zac(0:Ntor))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric coordinate axis Z cosine Fourier          &
          & coefficients")

      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Rbc", dset_id, Rbc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric boundary R cosine Fourier coefficients")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Zbs", dset_id, Zbs(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric boundary Z   sine Fourier coefficients")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Rbs", dset_id, Rbs(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric boundary R   sine Fourier coefficients")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Zbc", dset_id, Zbc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric boundary Z cosine Fourier coefficients")

      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Rwc", dset_id, Rwc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric boundary R cosine Fourier coefficients of wall")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Zws", dset_id, Zws(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric boundary Z   sine Fourier coefficients of wall")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Rws", dset_id, Rws(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric boundary R   sine Fourier coefficients of  &
          &         wall")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Zwc", dset_id, Zwc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric boundary Z cosine Fourier coefficients of  &
          &         wall")

      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Vns", dset_id, Vns(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric normal field   sine Fourier coefficients at    &
          &       boundary vacuum component")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Bns", dset_id, Bns(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "stellarator symmetric normal field   sine Fourier coefficients at    &
          &       boundary plasma component")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Vnc", dset_id, Vnc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric normal field cosine Fourier coefficients at&
          &           boundary vacuum component")
      call hdf5_write_real_2D_leave_open(grpInputPhysics, "Bnc", dset_id, Bnc(-Ntor:Ntor, -Mpol:Mpol))
      call hdf5_add_dataset_description(dset_id, "non-stellarator symmetric normal field cosine Fourier coefficients at&
          &           boundary plasma component")

      call hdf5_write_real_leave_open(grpInputPhysics, "mupftol", dset_id, [mupftol])
      call hdf5_add_dataset_description(dset_id, "mupftol")
      call hdf5_write_integer_leave_open(grpInputPhysics, "mupfits", dset_id, [mupfits])
      call hdf5_add_dataset_description(dset_id, "mupfits")

      call hdf5_close_group(grpInputPhysics)

      ! the following variables constitute the namelist/numericlist/: note that all variables in namelist need to be
      ! broadcasted in readin
      ! they go into ext.h5/input/numerics

      call hdf5_define_group(grpInput, "numerics", grpInputNumerics)

      call hdf5_write_integer(grpInputNumerics, "Linitialize", [Linitialize])
      call hdf5_write_integer(grpInputNumerics, "Ndiscrete", [Ndiscrete])
      call hdf5_write_integer(grpInputNumerics, "Nquad", [Nquad])
      call hdf5_write_integer(grpInputNumerics, "iMpol", [iMpol])
      call hdf5_write_integer(grpInputNumerics, "iNtor", [iNtor])
      call hdf5_write_integer(grpInputNumerics, "Lsparse", [Lsparse])
      call hdf5_write_integer(grpInputNumerics, "Lsvdiota", [Lsvdiota])
      call hdf5_write_integer(grpInputNumerics, "Lrzaxis", [Lrzaxis])
      call hdf5_write_integer(grpInputNumerics, "Ntoraxis", [Ntoraxis])

      call hdf5_close_group(grpInputNumerics)

      ! the following variables constitute the namelist/globallist/: note that all variables in namelist need to be
      ! broadcasted in readin
      ! they go into ext.h5/input/global

      call hdf5_define_group(grpInput, "global", grpInputGlobal)

      call hdf5_write_real(grpInputGlobal, "escale", [escale])
      call hdf5_write_real(grpInputGlobal, "opsilon", [opsilon])
      call hdf5_write_real(grpInputGlobal, "pcondense", [pcondense])
      call hdf5_write_real(grpInputGlobal, "epsilon", [epsilon])
      call hdf5_write_real(grpInputGlobal, "wpoloidal", [wpoloidal])
      call hdf5_write_real(grpInputGlobal, "upsilon", [upsilon])
      call hdf5_write_real(grpInputGlobal, "gBntol", [gBntol])
      call hdf5_write_real(grpInputGlobal, "gBnbld", [gBnbld])
      call hdf5_write_real(grpInputGlobal, "vcasingeps", [vcasingeps])
      call hdf5_write_real(grpInputGlobal, "vcasingtol", [vcasingtol])
      call hdf5_write_integer(grpInputGlobal, "vcasingits", [vcasingits])
      call hdf5_write_integer(grpInputGlobal, "vcasingper", [vcasingper])
      call hdf5_write_integer(grpInputGlobal, "vcNt", [vcNt])
      call hdf5_write_integer(grpInputGlobal, "vcNz", [vcNz])
      call hdf5_write_integer(grpInputGlobal, "cteta", [cteta])
      call hdf5_write_integer(grpInputGlobal, "czeta", [czeta])

      call hdf5_close_group(grpInputGlobal)

      ! the following variables constitute the namelist/diagnosticslist/: note that all variables in namelist need to
      ! be broadcasted in readin
      ! they go into ext.h5/input/diagnostics

      call hdf5_define_group(grpInput, "diagnostics", grpInputDiagnostics)

      call hdf5_write_real(grpInputDiagnostics, "odetol", [odetol])
      call hdf5_write_integer(grpInputDiagnostics, "nPpts", [nPpts])
      call hdf5_write_integer(grpInputDiagnostics, "Ppts", [int(Ppts)])
      call hdf5_write_integer(grpInputDiagnostics, "nPtrj", nPtrj(1:Mvol))
      call hdf5_write_logical(grpInputDiagnostics, "LHevalues", [LHevalues])
      call hdf5_write_logical(grpInputDiagnostics, "LHevectors", [LHevectors])
      call hdf5_write_logical(grpInputDiagnostics, "LHmatrix", [LHmatrix])
      call hdf5_write_logical(grpInputDiagnostics, "Ltransform", [Ltransform])
      call hdf5_write_integer(grpInputDiagnostics, "Lcheck", [Lcheck])
      call hdf5_write_integer(grpInputDiagnostics, "Lerrortype", [Lerrortype])
      call hdf5_write_integer(grpInputDiagnostics, "Ngrid", [Ngrid])

      call hdf5_close_group(grpInputDiagnostics)

      call hdf5_close_group(grpInput)

    end if

  end subroutine mirror_input_to_outfile

  subroutine write_grid
    use constants_mod
    use allglobal_mod, only: myid, ijreal, ijimag, jireal, Nt, Nz, Ntz, Mvol, pi2nfp, ivol, mn, Node,&
        & Lcoordinatesingularity, Lplasmaregion, Lvacuumregion, Rij, Zij, sg, inv_jac
    use input_list_mod, only: Lrad, Igeometry, Nvol, Ngrid, rtor, rpol
    use precisions_mod, only: rkind

    implicit none

    integer(hid_t) :: grpGrid
    integer :: sumLrad, alongLrad, Ngrid_local, Ngrid_sum
    integer :: vvol, ii, jj, kk, jk, Lcurvature
    real(rkind) :: lss, teta, zeta, st(1:Node), Bst(1:Node), gBzeta
    real(rkind), allocatable :: Rij_grid(:, :), Zij_grid(:, :), sg_grid(:, :), ijreal_grid(:, :), ijimag_grid(:, :),&
        & jireal_grid(:, :)

    if (myid == 0 .and. .not.skip_write) then

      ijreal(1:Ntz) = 0.0_rkind
      ijimag(1:Ntz) = 0.0_rkind
      jireal(1:Ntz) = 0.0_rkind

      call hdf5_define_group(file_id, "grid", grpGrid)

      ! Igeometry already is in input, Mvol already is in output
      call hdf5_write_integer(grpGrid, "Nt", [Nt])
      call hdf5_write_integer(grpGrid, "Nz", [Nz])
      call hdf5_write_integer(grpGrid, "Ntz", [Ntz])
      call hdf5_write_real(grpGrid, "pi2nfp", [pi2nfp])

      ! combine all radial parts into one dimension as Lrad values can be different for different volumes
      if (Ngrid < 0) then
        sumLrad = sum(Lrad(1:Mvol) + 1)
      else
        sumLrad = (Ngrid + 1) * Mvol
      end if

      call sallocate(Rij_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)
      call sallocate(Zij_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)
      call sallocate(sg_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)
      call sallocate(ijreal_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)
      call sallocate(ijimag_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)
      call sallocate(jireal_grid, 1, sumLrad, 1, Ntz, 0.0_rkind)

      Ngrid_sum = 0

      do vvol = 1, Mvol
        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

        if (Ngrid < 0) then
          Ngrid_local = Lrad(vvol)! default
        else
          Ngrid_local = Ngrid
        end if
        if (Ngrid_local == 0) cycle ! nothing to output

        do ii = 0, Ngrid_local ! sub-grid
          lss = ii * 2.0_rkind / Ngrid_local - 1.0_rkind
          if (Lcoordinatesingularity .and. ii == 0) then
            Lcurvature = 0 ! Jacobian is not defined
          else
            Lcurvature = 1 ! compute Jacobian
          end if

          call compute_coordinates(vvol, lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! only Rij(0,:) and Zij(0,:) are required: Rmn & Zmn are available

          alongLrad = Ngrid_sum + ii + 1

          Rij_grid(alongLrad, 1:Ntz) = Rij(1:Ntz, 0, 0)
          Zij_grid(alongLrad, 1:Ntz) = Zij(1:Ntz, 0, 0)
          sg_grid(alongLrad, 1:Ntz) = sg(1:Ntz, 0)

          if (Lcurvature == 1) then

            select case (Igeometry)

            case (3)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = [lss, teta]
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta /&
                      & sg(jk, 0)! BR
                  ijimag(jk) = (1.0_rkind) * gBzeta * inv_jac(jk)! Bp
                  jireal(jk) = (Zij(jk, 1, 0) * Bst(1) + Zij(jk, 2, 0) * Bst(2) + Zij(jk, 3, 0) * 1.0_rkind) * gBzeta /&
                      & sg(jk, 0)! BZ
                end do
              end do

            case (1)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = [lss, teta]
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta&
                      &  * inv_jac(jk)! BR
                  ijimag(jk) = (rpol) * gBzeta * inv_jac(jk)! Bzeta
                  jireal(jk) = (+rtor * Bst(2)) * gBzeta * inv_jac(jk)! Btheta
                end do
              end do

            case (2)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = [lss, teta]
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta&
                      &  * inv_jac(jk)! BR
                  ijimag(jk) = (1.0_rkind) * gBzeta * inv_jac(jk)! Bp
                  jireal(jk) = (Bst(2)) * gBzeta  * inv_jac(jk)! BZ
                end do
              end do

            end select !Igeometry
          end if

          ijreal_grid(alongLrad, 1:Ntz) = ijreal(1:Ntz)
          ijimag_grid(alongLrad, 1:Ntz) = ijimag(1:Ntz)
          jireal_grid(alongLrad, 1:Ntz) = jireal(1:Ntz)

        end do

        Ngrid_sum = Ngrid_sum + Ngrid_local + 1 ! offset for storing data

      end do

      call hdf5_write_real_2D(grpGrid, "Rij", Rij_grid)
      call hdf5_write_real_2D(grpGrid, "Zij", Zij_grid)
      call hdf5_write_real_2D(grpGrid, "sg", sg_grid)
      call hdf5_write_real_2D(grpGrid, "BR", ijreal_grid)
      call hdf5_write_real_2D(grpGrid, "Bp", ijimag_grid)
      call hdf5_write_real_2D(grpGrid, "BZ", jireal_grid)

      deallocate(Rij_grid)
      deallocate(Zij_grid)
      deallocate(sg_grid)
      deallocate(ijreal_grid)
      deallocate(ijimag_grid)
      deallocate(jireal_grid)

      call hdf5_close_group(grpGrid)

    end if

  end subroutine write_grid

  subroutine init_flt_output(numTrajTotal)
    use allglobal_mod, only: Nz, Mvol
    use input_list_mod, only: nPpts
    use precisions_mod, only: rkind

    implicit none

    integer, intent(in) :: numTrajTotal ! total number of trajectories
    integer(HSIZE_T), dimension(rankP) :: dims_traj ! Dataset dimensions.
    integer(HSIZE_T), dimension(rankP) :: length ! Dataset dimensions.

    if (myid == 0 .and. .not.skip_write) then

      ! create Poincare group in HDF5 file
      call hdf5_define_group(file_id, "poincare", grpPoincare)

      dims_traj = [Nz, nPpts, numTrajTotal] ! dimensions for whole Poincare dataset
      length = [Nz, nPpts, 1] ! which is written in these slice lengths

      ! Create the data space for the  dataset.
      call h5screate_simple_f(rankP, dims_traj, filespace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, dims_traj, filespace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, dims_traj, filespace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, dims_traj, filespace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(1, int([numTrajTotal], HSIZE_T), filespace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Create the dataset with default properties.
      call h5dcreate_f(grpPoincare, "t", H5T_NATIVE_DOUBLE, filespace_t, dset_id_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dcreate_f(grpPoincare, "s", H5T_NATIVE_DOUBLE, filespace_s, dset_id_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dcreate_f(grpPoincare, "R", H5T_NATIVE_DOUBLE, filespace_R, dset_id_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dcreate_f(grpPoincare, "Z", H5T_NATIVE_DOUBLE, filespace_Z, dset_id_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dcreate_f(grpPoincare, "success", H5T_NATIVE_integer, filespace_success, dset_id_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! filespaces can be closed as soon as datasets are created
      call h5sclose_f(filespace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Select hyperslab in the file.
      call h5dget_space_f(dset_id_t, filespace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dget_space_f(dset_id_s, filespace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dget_space_f(dset_id_R, filespace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dget_space_f(dset_id_Z, filespace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dget_space_f(dset_id_success, filespace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Each process defines dataset in memory and writes it to the hyperslab in the file.
      call h5screate_simple_f(rankP, length, memspace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, length, memspace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, length, memspace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankP, length, memspace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(1, int([1], HSIZE_T), memspace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! create rotational transform group in HDF5 file
      call hdf5_define_group(file_id, "transform", grpTransform)

      ! Create the data space for the  dataset.
      call h5screate_simple_f(rankT, int([2, Mvol], HSIZE_T), filespace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5screate_simple_f(rankT, int([numTrajTotal, 2], HSIZE_T), filespace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Create the dataset with default properties.
      call h5dcreate_f(grpTransform, "diotadxup", H5T_NATIVE_DOUBLE, filespace_diotadxup, dset_id_diotadxup,&
          & hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dcreate_f(grpTransform, "fiota", H5T_NATIVE_DOUBLE, filespace_fiota, dset_id_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! filespaces can be closed as soon as datasets are created
      call h5sclose_f(filespace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Select hyperslab in the file.
      call h5dget_space_f(dset_id_diotadxup, filespace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dget_space_f(dset_id_fiota, filespace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! Each process defines dataset in memory and writes it to the hyperslab in the file.
      call h5screate_simple_f(rankT, int([2, 1], HSIZE_T), memspace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

    end if

  end subroutine init_flt_output

  subroutine write_poincare(offset, data, success)
    use allglobal_mod, only: Nz
    use input_list_mod, only: nPpts
    use precisions_mod, only: rkind

    implicit none

    integer, intent(in) :: offset, success(:)
    real(rkind), intent(in) :: data(:, :, :)
    integer(hsize_t), dimension(3) :: length
    integer(HSIZE_T), dimension(2) :: dims_singleTraj ! dimensions of single trajectory data

    if (myid == 0 .and. .not.skip_write) then

      dims_singleTraj = [Nz, nPpts]
      length = [Nz, nPpts, 1]

      ! On entry, Fortran does not know that indexing in data is from 0 to Nz-1.
      ! Hence, use default indices 1:Nz in this routine
      call h5sselect_hyperslab_f(filespace_t, H5S_SELECT_SET_F, int([0, 0, offset], HSSIZE_T), length, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_t, H5T_NATIVE_DOUBLE, data(1, 1:Nz, 1:nPpts), dims_singleTraj, hdf5_ierr, file_space_id&
          & =filespace_t, mem_space_id=memspace_t)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sselect_hyperslab_f(filespace_s, H5S_SELECT_SET_F, int([0, 0, offset], HSSIZE_T), length, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_s, H5T_NATIVE_DOUBLE, data(2, 1:Nz, 1:nPpts), dims_singleTraj, hdf5_ierr, file_space_id&
          & =filespace_s, mem_space_id=memspace_s)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sselect_hyperslab_f(filespace_R, H5S_SELECT_SET_F, int([0, 0, offset], HSSIZE_T), length, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_R, H5T_NATIVE_DOUBLE, data(3, 1:Nz, 1:nPpts), dims_singleTraj, hdf5_ierr, file_space_id&
          & =filespace_R, mem_space_id=memspace_R)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sselect_hyperslab_f(filespace_Z, H5S_SELECT_SET_F, int([0, 0, offset], HSSIZE_T), length, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_Z, H5T_NATIVE_DOUBLE, data(4, 1:Nz, 1:nPpts), dims_singleTraj, hdf5_ierr, file_space_id&
          & =filespace_Z, mem_space_id=memspace_Z)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sselect_hyperslab_f(filespace_success, H5S_SELECT_SET_F, int([offset], HSSIZE_T), int([1], HSIZE_T),&
          & hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_success, H5T_NATIVE_integer, success, int([1], HSIZE_T), hdf5_ierr, file_space_id &
          &=filespace_success, mem_space_id=memspace_success)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

    end if

  end subroutine write_poincare

  subroutine write_transform(offset, length, lvol, diotadxup, fiota)
    use precisions_mod, only: rkind

    implicit none

    integer, intent(in) :: offset, length, lvol
    real(rkind), intent(in) :: diotadxup(:), fiota(:, :)

    if (myid == 0 .and. .not.skip_write) then

      call h5sselect_hyperslab_f(filespace_diotadxup, H5S_SELECT_SET_F, int([0, lvol - 1], HSSIZE_T), int([2, 1],&
          & HSSIZE_T), hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_diotadxup, H5T_NATIVE_DOUBLE, diotadxup, int([2, 1], HSSIZE_T), hdf5_ierr,&
          & file_space_id=filespace_diotadxup, mem_space_id=memspace_diotadxup)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! length of fiota piece to write here may change, so open and close memspace each time a new hyperslab is
      ! written
      call h5screate_simple_f(rankT, int([length, 2], HSIZE_T), memspace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sselect_hyperslab_f(filespace_fiota, H5S_SELECT_SET_F, int([offset, 0], HSSIZE_T), int([length, 2],&
          & HSSIZE_T), hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dwrite_f(dset_id_fiota, H5T_NATIVE_DOUBLE, fiota(1:length, 1:2), int([length, 2], HSSIZE_T), hdf5_ierr,&
          & file_space_id=filespace_fiota, mem_space_id=memspace_fiota)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      call h5sclose_f(memspace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

    end if

  end subroutine write_transform

  subroutine finalize_flt_output
    use precisions_mod, only: rkind

    implicit none

    if (myid == 0 .and. .not.skip_write) then

      ! close filespaces
      call h5sclose_f(filespace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(filespace_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! close dataspaces
      call h5sclose_f(memspace_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(memspace_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(memspace_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(memspace_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(memspace_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5sclose_f(memspace_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! memspace_fiota is re-opened/closed in each iteration (see write_transform)

      ! close datasets
      call h5dclose_f(dset_id_t, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_s, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_R, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_Z, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_success, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_diotadxup, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      call h5dclose_f(dset_id_fiota, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      ! close groups
      call hdf5_close_group(grpPoincare)
      call hdf5_close_group(grpTransform)

    end if

  end subroutine finalize_flt_output

  subroutine write_vector_potential(sumLrad, allAte, allAze, allAto, allAzo)
    use allglobal_mod, only: mn
    use precisions_mod, only: rkind

    implicit none

    integer, intent(in) :: sumLrad
    real(rkind), intent(in) :: allAte(:, :), allAze(:, :), allAto(:, :), allAzo(:, :)
    integer(hid_t) :: grpVectorPotential

    if (myid == 0 .and. .not.skip_write) then

      call hdf5_define_group(file_id, "vector_potential", grpVectorPotential)

      call hdf5_write_real_2D(grpVectorPotential, "Ate", allAte(1:sumLrad, 1:mn))
      call hdf5_write_real_2D(grpVectorPotential, "Aze", allAze(1:sumLrad, 1:mn))
      call hdf5_write_real_2D(grpVectorPotential, "Ato", allAto(1:sumLrad, 1:mn))
      call hdf5_write_real_2D(grpVectorPotential, "Azo", allAzo(1:sumLrad, 1:mn))

      call hdf5_close_group(grpVectorPotential)

    end if

  end subroutine write_vector_potential

  subroutine write_stability(ohessian, evalr, evali, evecr, NGdof)
    use input_list_mod, only: LHevalues, LHevectors, LHmatrix
    use precisions_mod, only: rkind

    implicit none

    integer, intent(in) :: NGdof
    real(rkind), intent(in) :: ohessian(:, :), evalr(:), evali(:), evecr(:, :)
    integer(hid_t) :: grpStability

    if (.not.LHevalues .and. .not.LHevectors .and. .not.LHmatrix) return

    if (myid == 0 .and. .not.skip_write) then

      call hdf5_define_group(file_id, "stability", grpStability)

      if (LHmatrix) then
        call hdf5_write_real_2D(grpStability, "ohessian", ohessian(1:NGdof, 1:NGdof))
      end if

      if (LHevectors) then
        call hdf5_write_real_2D(grpStability, "evecr", evecr(1:NGdof, 1:NGdof))
      end if

      if (LHevalues) then
        call hdf5_write_real(grpStability, "evalr", evalr(1:NGdof))
        call hdf5_write_real(grpStability, "evali", evali(1:NGdof))
      end if

      call hdf5_close_group(grpStability)

    end if

  end subroutine write_stability

  subroutine hdfint
    use io_mod, only: ounit
    use input_list_mod
    use allglobal_mod, only: Mvol, mn, vvolume, iVns, iBns, iVnc, iBnc, lmns, IPDt, dlambdaout, lmns
    use precisions_mod, only: rkind

    implicit none

    integer :: Mrad
    real(rkind) :: tvolume

    integer(hid_t) :: grpOutput

    if (myid == 0 .and. .not.skip_write) then

      call hdf5_define_group(file_id, "output", grpOutput)

      call hdf5_write_real(grpOutput, "Vns", iVns(1:mn))!     stellarator symmetric normal field at boundary: vacuum
      !     component
      call hdf5_write_real(grpOutput, "Bns", iBns(1:mn))!     stellarator symmetric normal field at boundary: plasma
      !     component
      call hdf5_write_real(grpOutput, "Vnc", iVnc(1:mn))! non-stellarator symmetric normal field at boundary: vacuum
      ! component
      call hdf5_write_real(grpOutput, "Bnc", iBnc(1:mn))! non-stellarator symmetric normal field at boundary: plasma
      ! component

      call hdf5_write_integer(grpOutput, "mn", [mn])
      call hdf5_write_integer(grpOutput, "im", im(1:mn))
      call hdf5_write_integer(grpOutput, "in", in(1:mn))
      call hdf5_write_integer(grpOutput, "mns", [mns])
      call hdf5_write_integer(grpOutput, "ims", ims(1:mns))
      call hdf5_write_integer(grpOutput, "ins", ins(1:mns))
      call hdf5_write_integer(grpOutput, "Mvol", [Mvol])
      call hdf5_write_real_2D(grpOutput, "Rbc", iRbc(1:mn, 0:Mvol))
      call hdf5_write_real_2D(grpOutput, "Zbs", iZbs(1:mn, 0:Mvol))
      call hdf5_write_real_2D(grpOutput, "Rbs", iRbs(1:mn, 0:Mvol))
      call hdf5_write_real_2D(grpOutput, "Zbc", iZbc(1:mn, 0:Mvol))
      call hdf5_write_real(grpOutput, "BnsErr", [BnsErr])! already in /input/global
      call hdf5_write_real(grpOutput, "ForceErr", [ForceErr])
      call hdf5_write_real(grpOutput, "Ivolume", Ivolume(1:Mvol))
      call hdf5_write_real(grpOutput, "IPDt", IPDt(1:Mvol))

      ! the following quantites can be different from input value
      call hdf5_write_real(grpOutput, "adiabatic", adiabatic(1:Nvol))
      call hdf5_write_real(grpOutput, "helicity", helicity(1:Nvol))
      call hdf5_write_real(grpOutput, "mu", mu(1:Mvol))
      call hdf5_write_real(grpOutput, "tflux", tflux(1:Mvol))
      call hdf5_write_real(grpOutput, "pflux", pflux(1:Mvol))

      if (Lcheck == 1) then
        call hdf5_write_real_2D(grpOutput, "beltramierror", beltramierror(1:Mvol, 1:3))
      end if

      if (allocated(vvolume)) then ! why is it required to confirm that vvolume has been allocated

        tvolume = sum(vvolume(1:Nvol))
        call hdf5_write_real(grpOutput, "volume", [tvolume])

      else

        if (Wsphdf5) write(ounit, '("hdfint : ", 10x ," : myid=",i3," vvolume is not allocated")')myid

      end if

      Mrad = maxval(Lrad(1:Mvol))
      call hdf5_write_integer(grpOutput, "Mrad", [Mrad])
      call hdf5_write_real_3D(grpOutput, "TT", TT(0:Mrad, 0:1, 0:1))
      call hdf5_write_real_3D(grpOutput, "Btemn", Btemn(1:mn, 0:1, 1:Mvol))
      call hdf5_write_real_3D(grpOutput, "Bzemn", Bzemn(1:mn, 0:1, 1:Mvol))
      call hdf5_write_real_3D(grpOutput, "Btomn", Btomn(1:mn, 0:1, 1:Mvol))
      call hdf5_write_real_3D(grpOutput, "Bzomn", Bzomn(1:mn, 0:1, 1:Mvol))

      ! Write lambda_mn, Fourier harmonics or transformation to straight field line coordinates.
      call hdf5_write_real_3D(grpOutput, "lambdamn", dlambdaout(1:lmns, 1:Mvol, 0:1))

      call hdf5_write_integer(grpOutput, "lmns", [lmns])

      call hdf5_close_group(grpOutput)

    end if

  end subroutine hdfint

  subroutine finish_outfile
    ! Close all open HDF5 objects (we know of) and list any remaining still-open objects
    ! The goal should be to close all objects specifically!
    use precisions_mod, only: rkind

    implicit none

    integer(size_t) :: obj_count ! number of open HDF5 objects
    integer(size_t) :: num_objs ! number of still-open objects
    integer(hid_t), dimension(:), allocatable :: obj_ids ! still-open objects
    integer :: iObj
    integer(size_t) :: openLength
    character(len=:), allocatable :: openName
    integer(size_t), parameter :: dummySize = 1
    character(len=dummySize + 1) :: dummyName
    integer :: typeClass

    if (myid == 0 .and. .not.skip_write) then

      !! Erol: Convergence output is not written so we don't need to close it
      ! ! close objects related to convergence output
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! End access to the
      ! ! dataset and release resources used by it.
      ! call h5pclose_f(plist_id, hdf5_ierr)
      ! call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! close plist used
      ! ! for
      ! ! 'preserve' flag (does not show up in obj_count below)

      ! check whether we forgot to close some resources: only check for group, dataset and datatype (there is only one
      ! file and that should be still open...)
      call h5fget_obj_count_f(file_id, ior(H5F_OBJ_GROUP_F, ior(H5F_OBJ_DATASET_F, H5F_OBJ_DATATYPE_F)), obj_count,&
          & hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))

      if (obj_count > 0) then
        write(stderr, '("There are still ",i3," hdf5 objects open")')obj_count
        allocate(obj_ids(1:obj_count))

        ! groups
        call h5fget_obj_ids_f(file_id, H5F_OBJ_GROUP_F, obj_count, obj_ids, hdf5_ierr, num_objs)
        call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! get for open
        ! objects
        if (num_objs > 0) then
          write(stderr, '("There are still ",i3," HDF5 groups open:")')num_objs
          do iObj = 1, num_objs
            openLength = 0
            call h5iget_name_f(obj_ids(iObj), dummyName, dummySize, openLength, hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
            allocate(character(len=openLength + 1) :: openName)
            call h5iget_name_f(obj_ids(iObj), openName, openLength, openLength, hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
            write(stderr, *)openName
            deallocate(openName)

            call h5gclose_f(obj_ids(iObj), hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
          end do
        end if

        ! datasets
        call h5fget_obj_ids_f(file_id, H5F_OBJ_DATASET_F, obj_count, obj_ids, hdf5_ierr, num_objs)
        call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! get for open
        ! objects
        if (num_objs > 0) then
          write(stderr, '("There are still ",i3," HDF5 datasets open:")')num_objs
          do iObj = 1, num_objs
            openLength = 0
            call h5iget_name_f(obj_ids(iObj), dummyName, dummySize, openLength, hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
            allocate(character(len=openLength + 1) :: openName)
            call h5iget_name_f(obj_ids(iObj), openName, openLength, openLength, hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
            write(stderr, *)openName(1:openLength)
            deallocate(openName)

            call h5dclose_f(obj_ids(iObj), hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
          end do
        end if

        ! datatypes
        call h5fget_obj_ids_f(file_id, H5F_OBJ_DATATYPE_F, obj_count, obj_ids, hdf5_ierr, num_objs)
        call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! get for open
        ! objects
        if (num_objs > 0) then
          write(stderr, '("There are still ",i3," HDF5 datatypes open:")')num_objs
          do iObj = 1, num_objs
            call h5tget_class_f(obj_ids(iObj), typeClass, hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! determine
            ! class of open datatype
            if (typeClass == H5T_NO_CLASS_F) then
              write(stderr, *)"H5T_NO_CLASS_F"
            else if (typeClass == H5T_integer_F) then
              write(stderr, *)"H5T_integer_F"
            else if (typeClass == H5T_FLOAT_F) then
              write(stderr, *)"H5T_FLOAT_F"
            else if (typeClass == H5T_STRING_F) then
              write(stderr, *)"H5T_STRING_F"
            else if (typeClass == H5T_BITFIELD_F) then
              write(stderr, *)"H5T_BITFIELD_F"
            else if (typeClass == H5T_OPAQUE_F) then
              write(stderr, *)"H5T_OPAQUE_F"
            else if (typeClass == H5T_COMPOUND_F) then
              write(stderr, *)"H5T_COMPOUND_F"
            else if (typeClass == H5T_REFERENCE_F) then
              write(stderr, *)"H5T_REFERENCE_F"
            else if (typeClass == H5T_ENUM_F) then
              write(stderr, *)"H5T_ENUM_F"
            else if (typeClass == H5T_VLEN_F) then
              write(stderr, *)"H5T_VLEN_F"
            else if (typeClass == H5T_ARRAY_F) then
              write(stderr, *)"H5T_ARRAY_F"
            else
              write(stderr, *)"UNKNOWN TYPE!"
            end if

            call h5tclose_f(obj_ids(iObj), hdf5_ierr)
            call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))
          end do
        end if

        deallocate(obj_ids)
      end if ! (obj_count.gt.0)

      call h5fclose_f(file_id, hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! terminate access on
      ! output file
      call h5close_f(hdf5_ierr)
      call fatal(hdf5_ierr /= 0, __FILE__, "Error calling HDF5 function at " // num2str(__LINE__))! close Fortran
      ! interface to the HDF5 library

    end if

  end subroutine finish_outfile

end module sphdf5_mod
