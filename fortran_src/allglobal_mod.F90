! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module allglobal_mod
#include "mpi_precomp.h"
  USE_MPI

  use constants_mod, only: pi
  use errors_mod, only: fatal
#ifdef __INTEL_COMPILER
  ! The Intel versions of ftell and fseek are function and not subroutines. They thus need a different treatment.
  use ifport, only: ftell, fseek
#endif
  use io_mod, only: ounit, iunit, stderr
  use input_list_mod, only: enforce_stell_sym, Igeometry, Lfreebound, Lbnszero, phiedge, curtor, curpol, gamma, Nfp, Nvol, &
      Mpol, Ntor, Lrad, tflux, pflux, helicity, pscale, Ladiabatic, pressure, adiabatic, mu, Lconstraint, &
      pl, ql, pr, qr, iota, lp, lq, rp, rq, oita, mupftol, mupfits, Lreflect, rpol, rtor, Ivolume, nPtrj, &
      epsilon, escale, gBnbld, gBntol, iMpol, iNtor, &
      LautoinitBn, Lcheck, LHevalues, LHevectors, LHmatrix, Linitialize, Lrzaxis, Lsparse, &
      Lsvdiota, Lzerovac, mfreeits, Ndiscrete, nPpts, Nquad, Ntoraxis, odetol, opsilon, pcondense, Ppts, &
      upsilon, vcasingeps, vcasingits, vcasingper, vcasingtol, Wmacros, wpoloidal, Wreadin, Wwrtend, Zbs, Rbc, Zbs, Vns, Bns, &
      Vnc, Bnc, Rwc, Zws, Rws, Zwc, Rbs, Zbc, Isurf, Lerrortype, Ltransform, MNvol, Ngrid, MMpol, MNtor, physicslist, &
      allRZRZ, mmRZRZ, nnRZRZ, num_modes, &
      numericlist, globallist, diagnosticslist, screenlist, vcNt, vcNz, Lvcgrid, Lbdybnzero
  use memory_allocation_mod, only: sallocate, nallocate
  use mpi_collectives_mod, only: broadcast, init_mpi_collective_mod
  use precisions_mod, only: rkind, machprec, vsmall
  use typedefns_mod, only: subgrid, derivative

  implicit none(type, external)

  private
  public :: IsMyVolume, build_vector_potential, WhichCpuID, get_hidden, set_mpi_comm, wrtend, read_inputlists_from_file, &
      check_inputs, broadcast_inputs, write_spec_namelist

  integer, public :: myid !< MPI rank of current CPU
  integer, public :: ncpu !< number of MPI tasks
  integer, public :: IsMyVolumeValue !< flag to indicate if a CPU is operating on its assigned volume
  real(rkind), public :: cpus !< initial time
  MPI_COMM_TYPE, public :: MPI_COMM_SPEC !< SPECTRE MPI communicator

  logical, public :: skip_write = .false. ! flag to disable any HDF5-related calls

  real(rkind), public :: pi2nfp !< pi2/nfp
  real(rkind), public :: pi2pi2nfp !< pi2*pi2/nfp
  real(rkind), public :: pi2pi2nfphalf !< 2*pi*pi/nfp
  real(rkind), public :: pi2pi2nfpquart !< pi*pi/nf

  character(LEN=1000), public :: ext ! extension of input filename, i.e., "path/G3V01L1Fi.001" for an input file
  ! path/G3V01L1Fi.001.sp

  character(LEN=1000), public :: filename_h5 ! name of the output h5 file

  real(rkind), public :: ForceErr !< total force-imbalance
  real(rkind), public :: Energy !< MHD energy
  real(rkind), public :: BnsErr !< (in freeboundary) error in self-consistency of field on plasma boundary (Picard iteration)

  !< Toroidal pressure-driven current
  real(rkind), allocatable, public :: IPDt(:)
  real(rkind), allocatable, public :: IPDtDpf(:, :)

  integer, public :: Mvol !< total number of volumes (including the vacuum region in the case of free-boundary calculations)

  real(rkind), public :: total_pflux ! used when Lconstraint=3, Igeometry=1

  !> Local workspace for evaluation of Chebychev polynomials
  real(rkind), allocatable, public :: cheby(:, :)
  !> Local workspace for evaluation of Zernike polynomials
  real(rkind), allocatable, public :: zernike(:, :, :)

  !> Derivatives of Chebyshev polynomials at the inner and outer interfaces
  real(rkind), allocatable, public :: TT(:, :, :)
  !> Derivatives of Zernike   polynomials at the inner and outer interfaces
  real(rkind), allocatable, public :: RTT(:, :, :, :)

  !< \f$r^m\f$ term of Zernike polynomials at the origin
  real(rkind), allocatable, public :: RTM(:, :)

  !> Enhanced resolution for metric elements
  integer, public :: mne
  !> Enhanced poloidal mode numbers for metric elements
  integer, allocatable, public :: ime(:)
  !> Enhanced toroidal mode numbers for metric elements
  integer, allocatable, public :: ine(:)

  integer, public :: mns !< enhanced resolution for straight field line transformation
  integer, allocatable, public :: ims(:)!< enhanced poloidal mode numbers for straight field line transformation
  integer, allocatable, public :: ins(:)!< enhanced toroidal mode numbers for straight field line transformation

  integer, public :: lMpol !< what is this?
  integer, public :: lNtor !< what is this?
  integer, public :: sMpol !< what is this?
  integer, public :: sNtor !< what is this?

  !> Used to normalize NAG routines (which ones exactly where?)
  real(rkind), public :: xoffset = 1.0_rkind

  !> Used to indicate if Beltrami fields have been correctly constructed
  logical, allocatable, public :: ImagneticOK(:)

  real(rkind), allocatable, public :: beltramierror(:, :)!< to store the integral of |curlB-mu*B| computed by compute_beltrami_error

  integer, public :: mn !< total number of Fourier harmonics for coordinates/fields: calculated from Mpol, Ntor in readin()
  integer, allocatable, public :: im(:)!< poloidal mode numbers for Fourier representation
  integer, allocatable, public :: in(:)!< toroidal mode numbers for Fourier representation

  real(rkind), allocatable, public :: halfmm(:)!< I saw this already somewhere...

  real(rkind), public :: Rscale !< no idea
  real(rkind), allocatable, public :: psifactor(:, :)!< no idea
  real(rkind), allocatable, public :: inifactor(:, :)!< no idea

  !> Weight on force-imbalance harmonics: used in dforce()
  real(rkind), allocatable, public :: BBweight(:)

  !> Spectral condensation factors
  real(rkind), allocatable, public :: mmpp(:)

  real(rkind), allocatable, public :: iRbc(:, :)!< cosine R harmonics of interface surface geometry:     stellarator symmetric
  real(rkind), allocatable, public :: iZbs(:, :)!<   sine Z harmonics of interface surface geometry:     stellarator symmetric
  real(rkind), allocatable, public :: iRbs(:, :)!<   sine R harmonics of interface surface geometry: non-stellarator symmetric
  real(rkind), allocatable, public :: iZbc(:, :)!< cosine Z harmonics of interface surface geometry: non-stellarator symmetric

  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: iRij(:, :)
  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: iZij(:, :)
  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: dRij(:, :)
  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: dZij(:, :)
  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: tRij(:, :)
  !> Interface surface geometry: real space
  real(rkind), allocatable, public :: tZij(:, :)

  !> Sine harmonics of vacuum normal magnetic field on interfaces: stellarator symmetric
  real(rkind), allocatable, public :: iVns(:)
  !> Sine harmonics of plasma normal magnetic field on interfaces: stellarator symmetric
  real(rkind), allocatable, public :: iBns(:)
  !> Sosine harmonics of vacuum normal magnetic field on interfaces: non-stellarator symmetric
  real(rkind), allocatable, public :: iVnc(:)
  !> Cosine harmonics of plasma normal magnetic field on interfaces: non-stellarator symmetric
  real(rkind), allocatable, public :: iBnc(:)


  integer, public :: Nt !< discrete resolution along \f$\theta\f$ of grid in real space
  integer, public :: Nz !< discrete resolution along \f$\zeta\f$  of grid in real space
  integer, public :: Ntz !< discrete resolution: Ntz=Nt*Nz shorthand
  integer, public :: hNt !< discrete resolution: Ntz=Nt*Nz shorthand
  integer, public :: hNz !< discrete resolution: Ntz=Nt*Nz shorthand
  real(rkind), public :: soNtz !< 1.0_rkind / sqrt (1.0_rkind*Ntz)

  real(rkind), allocatable, public :: Rij(:, :, :)!< real-space grid: R
  real(rkind), allocatable, public :: Zij(:, :, :)!< real-space grid: Z
  real(rkind), allocatable, public :: sg(:, :)!< real-space grid: jacobian and its derivatives
  real(rkind), allocatable, public :: inv_jac(:)!< real-space grid: inverse jacobian
  real(rkind), allocatable, public :: guvij(:, :, :, :)!< real-space grid: metric elements
  real(rkind), allocatable, public :: gvuij(:, :, :)!< real-space grid: metric elements (?)
  real(rkind), allocatable, public :: guvijsave(:, :, :, :)!< what is this?

  integer, allocatable, public :: ki(:, :)!< identification of Fourier modes
  !> Identification of Fourier modes
  integer, allocatable, public :: kijs(:, :, :)
  !> Identification of Fourier modes
  integer, allocatable, public :: kija(:, :, :)

  integer, allocatable, public :: iotakkii(:)!< identification of Fourier modes
  integer, allocatable, public :: iotaksub(:, :)!< identification of Fourier modes
  integer, allocatable, public :: iotakadd(:, :)!< identification of Fourier modes
  integer, allocatable, public :: iotaksgn(:, :)!< identification of Fourier modes

  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: efmn(:)
  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: ofmn(:)

  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: cfmn(:)
  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: sfmn(:)
  !> Fourier harmonics: dummy workspace

  real(rkind), allocatable, public :: evmn(:)
  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: odmn(:)
  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: comn(:)
  !> Fourier harmonics: dummy workspace
  real(rkind), allocatable, public :: simn(:)

  real(rkind), allocatable, public :: ijreal(:)!< what is this ?
  real(rkind), allocatable, public :: ijimag(:)!< what is this ?
  real(rkind), allocatable, public :: jireal(:)!< what is this ?
  real(rkind), allocatable, public :: jiimag(:)!< what is this ?

  real(rkind), allocatable, public :: jkreal(:)!< what is this ?
  real(rkind), allocatable, public :: jkimag(:)!< what is this ?
  real(rkind), allocatable, public :: kjreal(:)!< what is this ?
  real(rkind), allocatable, public :: kjimag(:)!< what is this ?

  real(rkind), allocatable, public :: Bsupumn(:, :, :)!< tangential field on interfaces: \f$\theta\f$-component: required for
  !virtual casing construction of field
  real(rkind), allocatable, public :: Bsupvmn(:, :, :)!< tangential field on interfaces: \f$\zeta\f$ -component: required for
  !virtual casing construction of field

  real(rkind), allocatable, public :: goomne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: goomno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gssmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gssmno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gstmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gstmno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gszmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gszmno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gttmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gttmno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gtzmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gtzmno(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gzzmne(:, :)!< described in initialize_internal_arrays()
  real(rkind), allocatable, public :: gzzmno(:, :)!< described in initialize_internal_arrays()

  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DToocc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DToocs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DToosc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DTooss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TTsscc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TTsscs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TTsssc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TTssss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDstcc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDstcs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDstsc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDstss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDszcc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDszcs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDszsc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: TDszss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDttcc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDttcs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDttsc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDttss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDtzcc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDtzcs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDtzsc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDtzss(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDzzcc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDzzcs(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDzzsc(:, :, :, :)
  !>Volume-integrated Chebychev-metrics
  real(rkind), allocatable, public :: DDzzss(:, :, :, :)

  real(rkind), allocatable, public :: Tsc(:, :)!< what is this?
  real(rkind), allocatable, public :: Tss(:, :)!< what is this?
  real(rkind), allocatable, public :: Dtc(:, :)!< what is this?
  real(rkind), allocatable, public :: Dts(:, :)!< what is this?
  real(rkind), allocatable, public :: Dzc(:, :)!< what is this?
  real(rkind), allocatable, public :: Dzs(:, :)!< what is this?
  real(rkind), allocatable, public :: Ttc(:, :)!< what is this?
  real(rkind), allocatable, public :: Tzc(:, :)!< what is this?
  real(rkind), allocatable, public :: Tts(:, :)!< what is this?
  real(rkind), allocatable, public :: Tzs(:, :)!< what is this?

  !> \f$\delta \psi_{toroidal}\f$ in each annulus
  real(rkind), allocatable, public :: dtflux(:)
  !> \f$\delta \psi_{poloidal}\f$ in each annulus
  real(rkind), allocatable, public :: dpflux(:)

  !> Minimum poloidal length constraint weight
  real(rkind), allocatable, public :: sweight(:)

  !> Degrees of freedom in Beltrami fields in each annulus
  integer, allocatable, public :: NAdof(:)

  !> Magnetic vector potential cosine Fourier harmonics: stellarator-symmetric
  type(subgrid), allocatable, public :: Ate(:, :, :)
  !> Magnetic vector potential cosine Fourier harmonics: stellarator-symmetric
  type(subgrid), allocatable, public :: Aze(:, :, :)
  !> Magnetic vector potential   sine Fourier harmonics: non-stellarator-symmetric
  type(subgrid), allocatable, public :: Ato(:, :, :)
  !> Magnetic vector potential   sine Fourier harmonics: non-stellarator-symmetric
  type(subgrid), allocatable, public :: Azo(:, :, :)

  integer, allocatable, public :: Lma(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmb(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmc(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmd(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lme(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmf(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmg(:, :)!< Lagrange multipliers (?)
  integer, allocatable, public :: Lmh(:, :)!< Lagrange multipliers (?)

  real(rkind), allocatable, public :: Lmavalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmbvalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmcvalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmdvalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmevalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmfvalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmgvalue(:, :)!< what is this?
  real(rkind), allocatable, public :: Lmhvalue(:, :)!< what is this?

  ! NOT threadprivate: read inside live !$OMP regions (matrices_mod, chebyshev_mod); passed as an argument on the threaded path
  logical, public :: Lcoordinatesingularity !< set by \c lregion subroutine: true if inside the innermost volume
  logical, public :: Lplasmaregion !< set by \c lregion subroutine: true if inside the plasma region
  !$OMP THREADPRIVATE(Lplasmaregion)
  logical, public :: Lvacuumregion !< set by \c lregion subroutine: true if inside the vacuum region
  !$OMP THREADPRIVATE(Lvacuumregion)
  logical, public :: Lsavedguvij !< flag used in matrix free
  !$OMP THREADPRIVATE(Lsavedguvij)
  logical, public :: Localconstraint !< what is this?

  !> Energy and helicity matrices: quadratic forms
  real(rkind), allocatable, public :: dMA(:, :)
  !> Energy and helicity matrices: quadratic forms
  real(rkind), allocatable, public :: dMB(:, :)
  !> Energy and helicity matrices: quadratic forms
  real(rkind), allocatable, public :: dMD(:, :)

  real(rkind), allocatable, public :: dMG(:)!< what is this?

  !> This is allocated in dforce: used in solve_beltrami_system and construct_beltrami_field: and is passed to packab
  real(rkind), allocatable, public :: solution(:, :)

  !> Measured rotational transform on inner/outer interfaces for each volume: d(transform)/dx
  real(rkind), allocatable, public :: diotadxup(:, :, :)
  !> Measured toroidal and poloidal current on inner/outer interfaces for each volume: d(Itor,Gpol)/dx
  real(rkind), allocatable, public :: dItGpdxtp(:, :, :)
  !> Save initial guesses for iterative calculation of rotational-transform
  real(rkind), allocatable, public :: glambda(:, :, :, :)

  !> Number of independent degrees of freedom in angle transformation
  integer, public :: lmns

  real(rkind), allocatable, public :: dlambdaout(:, :, :)

  real(rkind), allocatable, public :: Bemn(:, :, :)!< force vector
  real(rkind), allocatable, public :: Iomn(:, :)!< force vector
  real(rkind), allocatable, public :: Somn(:, :, :)!< force vector
  real(rkind), allocatable, public :: Pomn(:, :, :)!< force vector

  real(rkind), allocatable, public :: Bomn(:, :, :)!< force vector
  real(rkind), allocatable, public :: Iemn(:, :)!< force vector
  real(rkind), allocatable, public :: Semn(:, :, :)!< force vector
  real(rkind), allocatable, public :: Pemn(:, :, :)!< force vector

  real(rkind), allocatable, public :: BBe(:)!< force vector (?)
  real(rkind), allocatable, public :: IIo(:)!< force vector (?)
  real(rkind), allocatable, public :: BBo(:)!< force vector (?)
  real(rkind), allocatable, public :: IIe(:)!< force vector (?)

  !> Covariant \f$\theta\f$ cosine component of the tangential field on interfaces: stellarator-symmetric
  real(rkind), allocatable, public :: Btemn(:, :, :)
  !> covariant \f$\zeta\f$  cosine component of the tangential field on interfaces: stellarator-symmetric
  real(rkind), allocatable, public :: Bzemn(:, :, :)
  !> covariant \f$\theta\f$   sine component of the tangential field on interfaces: non-stellarator-symmetric
  real(rkind), allocatable, public :: Btomn(:, :, :)
  !> covariant \f$\zeta\f$    sine component of the tangential field on interfaces: non-stellarator-symmetric
  real(rkind), allocatable, public :: Bzomn(:, :, :)

  !> Covariant field for Hessian computation
  real(rkind), allocatable, public :: Bloweremn(:, :)
  !> covariant field for Hessian computation
  real(rkind), allocatable, public :: Bloweromn(:, :)

  !> Geometrical degrees of freedom associated with each interface
  integer, public :: LGdof
  !> Total geometrical degrees of freedom
  integer, public :: NGdof

  !> Derivative of magnetic field w.r.t. geometry (?)
  real(rkind), allocatable, public :: dBBdRZ(:, :, :)

  !> Derivatives of B^2 at the interfaces wrt geometry
  real(rkind), allocatable, public :: dFFdRZ(:, :, :, :, :)
  !> Derivatives of B^2 at the interfaces wrt mu and dpflux
  real(rkind), allocatable, public :: dBBdmp(:, :, :, :)

  !> Derivatives of B^2 at the interfaces wrt geometry 2D Hessian
  real(rkind), allocatable, public :: HdFFdRZ(:, :, :, :, :)

  !> Derivatives of energy at the interfaces wrt geometry 3D Hessian
  real(rkind), allocatable, public :: denergydrr(:, :, :, :, :)

  !> Derivatives of energy at the interfaces wrt geometry 3D Hessian
  real(rkind), allocatable, public :: denergydzr(:, :, :, :, :)

  !> Derivatives of mu and dpflux wrt geometry at constant interface transform
  real(rkind), allocatable, public :: dmupfdx(:, :, :, :, :)

  !> Flag to indicate that force gradient matrix is allocated (?)
  logical, public :: force_ft_trunc_grad_alloc
  !> Force gradient matrix (?)
  real(rkind), allocatable, public :: force_ft_trunc_grad(:, :)

  !> Flag to indicate that 2D Hessian matrix is allocated (?)
  logical, public :: Lhessian2Dallocated = .false.
  !> Hessian 2D
  real(rkind), allocatable, public :: hessian2D(:, :)

  !> some precomputed cosines
  real(rkind), allocatable, public :: cosi(:, :)
  !> some precomputed sines
  real(rkind), allocatable, public :: sini(:, :)
  real(rkind), allocatable, public :: gteta(:)!< something related to \f$\sqrt g\f$ and \f$\theta\f$ ?
  real(rkind), allocatable, public :: gzeta(:)!< something related to \f$\sqrt g\f$ and \f$\zeta\f$ ?

  real(rkind), allocatable, public :: ajk(:)!< definition of coordinate axis

  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dRadR(:, :, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dRadZ(:, :, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dZadR(:, :, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dZadZ(:, :, :, :)

  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dRodR(:, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dRodZ(:, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dZodR(:, :, :)
  !> Derivatives of coordinate axis
  real(rkind), allocatable, public :: dZodZ(:, :, :)

  integer, allocatable, public :: djkp(:, :)!< for calculating cylindrical volume
  integer, allocatable, public :: djkm(:, :)!< for calculating cylindrical volume

  real(rkind), allocatable, public :: lBBintegral(:)!< B.B integral
  real(rkind), allocatable, public :: lABintegral(:)!< A.B integral

  real(rkind), allocatable, public :: vvolume(:)!< volume integral of \f$\sqrt g\f$: computed in volume
  real(rkind), public :: dvolume !< derivative of volume w.r.t. interface geometry
  !$OMP THREADPRIVATE(dvolume)

  !> labels volume: some subroutines (called by NAG) are fixed argument list but require the volume label
  integer, public :: ivol

  !> internal copy of Nquad
  integer, allocatable, public :: Iquad(:)

  real(rkind), allocatable, public :: gaussianweight(:, :)!<   weights for Gaussian quadrature
  real(rkind), allocatable, public :: gaussianabscissae(:, :)!< abscissae for Gaussian quadrature

  type(derivative), public :: dBdX !< \f${\rm d}\mathbf{B}/{\rm d}\mathbf{X}\f$ (?)
  !$OMP THREADPRIVATE(dBdX)

  !> Labels position
  integer, public :: globaljk
  !> Computational boundary: position
  real(rkind), allocatable, public :: Dxyz(:, :)
  !> Computational boundary: normal
  real(rkind), allocatable, public :: Nxyz(:, :)
  !> Plasma boundary: surface current
  real(rkind), allocatable, public :: Jxyz(:, :)
  !> Plasma boundary: position
  real(rkind), allocatable, public :: Pbxyz(:, :)

  integer, public :: prevcstride !< previous virtual casing stride for casinggrid (only relevant when Lvcgrid=1)

  real(rkind), public :: virtualcasingfactor = -1.0_rkind / (4.0_rkind * pi) !< this agrees with diagno

  integer, public :: nfreeboundaryiterations !< number of free-boundary iterations already performed

  !> Best to make this global for consistency between calling and called routines
  integer, parameter, public :: Node = 2

  !> Flag to indicate that this is the first free-boundary iteration
  logical, public :: first_free_bound = .false.

  !> Flag to indicate whether Beltrami field in the vacuum volume will be found consistently
  !> Essentially picard iteration on the dfp100 call for the last volume
  !> Relevant only with Lfreebound=.true.
  logical, public :: update_bnorm_inbeltrami = .false.

  !> Store external mesh for the field
  real, allocatable, public :: Remn_ext(:, :, :)
  real, allocatable, public :: Romn_ext(:, :, :)
  real, allocatable, public :: Zemn_ext(:, :, :)
  real, allocatable, public :: Zomn_ext(:, :, :)

  logical, public :: use_ext_mesh = .false.

contains

  subroutine build_vector_potential(lvol, iocons, aderiv, Lcoordinatesingularity, efmn, ofmn, cfmn, sfmn)
    integer, intent(in) :: lvol
    integer, intent(in) :: iocons
    integer, intent(in) :: aderiv
    logical, intent(in) :: Lcoordinatesingularity
    ! Vector-potential harmonics output; callers pass the module globals (serial) or thread-local arrays.
    real(rkind), intent(out) :: efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn)

    integer :: ii, & ! Loop index on Fourier harmonics
        ll, & ! Loop index on radial resolution
        mi ! Poloidal mode number

    efmn(1:mn) = 0.0_rkind
    sfmn(1:mn) = 0.0_rkind
    cfmn(1:mn) = 0.0_rkind
    ofmn(1:mn) = 0.0_rkind

    do ii = 1, mn ! loop over Fourier harmonics

      if (Lcoordinatesingularity) then
        mi = im(ii)
        do ll = mi, Lrad(lvol), 2 ! loop over Zernike polynomials: Lrad is the radial resolution
          efmn(ii) = efmn(ii) + Ate(lvol, aderiv, ii)%s(ll) * RTT(ll, mi, iocons, 1) * 1.0_rkind / 2.0_rkind
          cfmn(ii) = cfmn(ii) + Aze(lvol, aderiv, ii)%s(ll) * RTT(ll, mi, iocons, 1) * 1.0_rkind / 2.0_rkind
          if (.not.enforce_stell_sym) then
            ofmn(ii) = ofmn(ii) + Ato(lvol, aderiv, ii)%s(ll) * RTT(ll, mi, iocons, 1) * 1.0_rkind / 2.0_rkind
            sfmn(ii) = sfmn(ii) + Azo(lvol, aderiv, ii)%s(ll) * RTT(ll, mi, iocons, 1) * 1.0_rkind / 2.0_rkind
          end if
        end do
      else
        do ll = 0, Lrad(lvol)! loop over Chebyshev polynomials: Lrad is the radial resolution
          efmn(ii) = efmn(ii) + Ate(lvol, aderiv, ii)%s(ll) * TT(ll, iocons, 1)! aderiv labels deriv. wrt mu, pflux
          cfmn(ii) = cfmn(ii) + Aze(lvol, aderiv, ii)%s(ll) * TT(ll, iocons, 1)
          if (.not.enforce_stell_sym) then
            ofmn(ii) = ofmn(ii) + Ato(lvol, aderiv, ii)%s(ll) * TT(ll, iocons, 1)
            sfmn(ii) = sfmn(ii) + Azo(lvol, aderiv, ii)%s(ll) * TT(ll, iocons, 1)
          end if
        end do
      end if ! Lcoordinatesingularity
    end do
  end subroutine build_vector_potential

  ! Use this routine to run SPECTRE on a different MPI communicator.
  ! It takes care to re-assign the rank and size information
  ! that is used internally in SPECTRE.
  subroutine set_mpi_comm(comm)
    integer, intent(in) :: comm
    integer :: ierr

    ! MPI_COMM_SPEC is the global variable for the SPECTRE communicator.
#ifdef NO_MPI_F08
    MPI_COMM_SPEC = comm
#else
    MPI_COMM_SPEC%mpi_val = comm
#endif


    myid = 0
    ncpu = 1

    ierr = 0
    call MPI_COMM_RANK(MPI_COMM_SPEC, myid, ierr)
    if (ierr /= 0) write(stderr, *)"error in call to MPI_COMM_RANK"

    ierr = 0
    call MPI_COMM_SIZE(MPI_COMM_SPEC, ncpu, ierr)
    if (ierr /= 0) write(stderr, *)"error in call to MPI_COMM_SIZE"

    call init_mpi_collective_mod(MPI_COMM_SPEC)
  end subroutine set_mpi_comm

  pure function get_hidden(ext)result(hidden_ext)
    character(len=1000), intent(in) :: ext
    ! ext with a "." prefix added to the basename "path/.G3V01L1Fi.001" for an input file path/G3V01L1Fi.001.sp
    character(LEN=1000) :: hidden_ext
    integer :: basename_start_index

    ! Prepare the "hidden" ext filepath that has a "." prefix.
    ! Split ext into directory path and basename using INDEX function, then concatenate them again with a "." inbetween
    basename_start_index = index(ext, '/', .true.)
    ! folder + . + filename
    hidden_ext = trim(ext(1:basename_start_index)) // "." // trim(ext(basename_start_index + 1:))
  end function get_hidden

  subroutine read_inputlists_from_file()
    use memory_allocation_mod, only: sallocate
    real(rkind) :: cput

    logical :: Lspexist
    integer :: filepos, seek_status, instat, idx_mode

    character(len=1000) :: line

    integer :: mm, nn, MNMAX
    real(rkind), allocatable :: RZRZ(:, :)! local array used for reading interface Fourier harmonics from file

    inquire(file=trim(ext) // ".sp", exist=Lspexist)! check if file exists
    call fatal(.not.Lspexist, __FILE__, "the input file does not exist ")
    ! if not, abort

    open(iunit, file=trim(ext) // ".sp", status="old")

    instat = 0 ! initially, no error

    ! read namelists one after another
    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : reading physicslist from ext.sp ")')cput - cpus
    end if

    read(iunit, physicslist, iostat=instat)
    if (instat /= 0) then
      ! help to debug invalid inputs:
      ! re-read last line that lead to error and print it to screen
      backspace(iunit)
      read(iunit, fmt = '(A)')line
      write(stderr, '(A)')'Invalid line in physicslist: ' // trim(line)
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : read    physicslist from ext.sp")')cput - cpus
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : reading numericlist from ext.sp")')cput - cpus
    end if

    read(iunit, numericlist, iostat=instat)
    if (instat /= 0) then
      ! help to debug invalid inputs:
      ! re-read last line that lead to error and print it to screen
      backspace(iunit)
      read(iunit, fmt = '(A)')line
      write(stderr, '(A)')'Invalid line in numericlist: ' // trim(line)
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : read numericlist from ext.sp")')cput - cpus
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : reading globallist from ext.sp")')cput - cpus
    end if

    read(iunit, globallist, iostat=instat)
    if (instat /= 0) then
      ! help to debug invalid inputs:
      ! re-read last line that lead to error and print it to screen
      backspace(iunit)
      read(iunit, fmt = '(A)')line
      write(stderr, '(A)')'Invalid line in globallist: ' // trim(line)
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : read globallist from ext.sp")')cput - cpus
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : reading diagnosticslist from ext.sp")')cput - cpus
    end if

    read(iunit, diagnosticslist, iostat=instat)
    if (instat /= 0) then
      ! help to debug invalid inputs:
      ! re-read last line that lead to error and print it to screen
      backspace(iunit)
      read(iunit, fmt = '(A)')line
      write(stderr, '(A)')'Invalid line in diagnosticslist: ' // trim(line)
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : read diagnosticslist from ext.sp")')cput - cpus
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : reading screenlist from ext.sp")')cput - cpus
    end if

    read(iunit, screenlist, iostat=instat)
    if (instat /= 0) then
      ! help to debug invalid inputs:
      ! re-read last line that lead to error and print it to screen
      backspace(iunit)
      read(iunit, fmt = '(A)')line
      write(stderr, '(A)')'Invalid line in screenlist: ' // trim(line)
    end if

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : read screenlist from ext.sp")')cput - cpus
    end if

    ! At this point, the input namelists are read.
    ! It remains to read the initial guess for the interface geometry,
    ! which follows after the namelists in the input file.

    ! need to reset status flag for below logic to work
    instat = 0

    num_modes = 0

    MNMAX = MNtor + 1 + MMpol * (2 * MNtor + 1)
    if (allocated(mmRZRZ)) deallocate(mmRZRZ, nnRZRZ, allRZRZ)

    call nallocate(mmRZRZ, 1, MNMAX)
    call nallocate(nnRZRZ, 1, MNMAX)
    call nallocate(allRZRZ, 1, 4, 1, Nvol, 1, MNMAX)

    if (Linitialize <= 0) then

      ! duplicate of checks required for below code
      call fatal(Nvol < 1 .or. Nvol > MNvol, __FILE__, "invalid Nvol: may need to recompile with higher MNvol ")

      call sallocate(RZRZ, 1, 4, 1, Nvol, 0.0_rkind)! temp array for reading input

      ! determine how many modes are specified by reading them once
#ifdef __INTEL_COMPILER
      filepos = ftell(iunit) + 1
#else
      call ftell(iunit, filepos)
#endif
      do ! will read in Fourier harmonics until the end of file is reached
        read(iunit, *, iostat=instat)mm, nn, RZRZ(1:4, 1:Nvol)!if change of angle applies, transformation assumes
        !m>=0 and for m=0 only n>=0
        if (instat /= 0) exit

        num_modes = num_modes + 1
      end do

      ! rewind file to reset EOF flag
      rewind(iunit)

      ! seek back to (start of modes) == (end of input namelists)
#ifdef __INTEL_COMPILER
      seek_status = fseek(iunit, filepos, 0)
#else
      call fseek(iunit, filepos, 0, seek_status)
#endif
      call fatal(seek_status /= 0, __FILE__, "failed to seek to end of input namelists ")

      ! now allocate arrays and read...
      ! Need to free memory, in case initialize_internal_arrays() called multiple times via python wrappers
      do idx_mode = 1, num_modes
        read(iunit, *, iostat=instat)mmRZRZ(idx_mode), nnRZRZ(idx_mode), allRZRZ(1:4, 1:Nvol, idx_mode)
      end do

      ! no need for temporary RZRZ anymore
      deallocate(RZRZ)

    end if ! Linitialize .le. 0

    close(iunit)

  end subroutine read_inputlists_from_file ! read_inputlists_from_file

  subroutine write_spec_namelist()
    logical :: exist
    character(LEN=100), parameter :: example = 'example.sp'

    if (myid == 0) then
      inquire(file=trim(example), EXIST=exist)! inquire if inputfile existed
      call fatal(exist, __FILE__, "example input file example.sp already existed ")

      open(iunit, file=trim(example), status='unknown', action='write')
      write(iunit, physicslist)
      write(iunit, numericlist)
      write(iunit, globallist)
      write(iunit, diagnosticslist)
      write(iunit, screenlist)
      close(iunit)
    end if

    return
  end subroutine write_spec_namelist

  subroutine check_inputs()
    real(rkind) :: cput

    integer :: vvol
    real(rkind) :: toroidalflux, toroidalcurrent

    Mvol = Nvol + Lfreebound ! this is just for screen output and initial check: true assignment of Mvol appears
    ! outside if( myid.eq.0 ) then

    write(ounit, '("readin : ", 10x ," : ")')

    cput = MPI_Wtime()

    write(ounit, 1010)cput - cpus, Igeometry, enforce_stell_sym, Lreflect, Lbdybnzero
    write(ounit, 1011)Lfreebound, phiedge, curtor, curpol
    write(ounit, 1012)gamma
    write(ounit, 1013)Nfp, Nvol, Mvol, Mpol, Ntor
    write(ounit, 1014)pscale, Ladiabatic, Lconstraint, mupftol, mupfits
    write(ounit, 1015)Lrad(1:min(Mvol, 32))

    1010 format("readin : ", f10.2, " : Igeometry=", i3, " ; enforce_stell_sym=", l3, " ; Lreflect=", i3, " ; Lbdybnzero=", l2, " ")
    1011 format("readin : ", 10x, " : Lfreebound=", i3, " ; phiedge=", es23.15, " ; curtor=", es23.15, " ; curpol=", es23.15, " ")
    1012 format("readin : ", 10x, " : gamma=", es23.15, " ")
    1013 format("readin : ", 10x, " : Nfp=", i3, " ; Nvol=", i3, " ; Mvol=", i3, " ; Mpol=", i3, " ; Ntor=", i3, " ")
    1014 format("readin : ", 10x, " : pscale=", es13.5, " ; Ladiabatic=", i2, " ; Lconstraint=", i3, " ; mupf: tol,its=", es10.2, ",", i4, " ")
    1015 format("readin : ", 10x, " : Lrad = ", 257(i2, ",", :))

    call fatal(Igeometry < 1 .or. Igeometry > 3, __FILE__, "invalid geometry ")

    call fatal(Nfp <= 0, __FILE__, "invalid Nfp ")

    call fatal(Mpol < 0 .or. Mpol > MMpol, __FILE__, "invalid poloidal resolution: may need to recompile with higher   &
        &      MMpol ")

    call fatal(Ntor < 0 .or. Ntor > MNtor, __FILE__, "invalid toroidal resolution: may need to recompile with higher   &
        &      MNtor ")

    call fatal(Nvol < 1 .or. Nvol > MNvol, __FILE__, "invalid Nvol: may need to recompile with higher MNvol ")

    call fatal(mupftol <= 0.0_rkind, __FILE__, "mupftol is too small ")

    call fatal(abs(1.0_rkind + gamma) < vsmall, __FILE__, "1+gamma appears in denominator in dforce ")
    !< \todo Please check this: SRH:
    call fatal(abs(1.0_rkind - gamma) < vsmall, __FILE__, "1-gamma appears in denominator in fu00aa ")
    !< \todo Please check this: SRH:

    call fatal(Lconstraint < -2 .or. Lconstraint > 3, __FILE__, "Illegal Lconstraint ")
    call fatal(Lconstraint == -2 .and. Nvol /= 1, __FILE__, " Lconstraint=-2 only for single-volume calculation ")
    call fatal(.not.Lbdybnzero .and. (Lconstraint /=-2 .or. Lfreebound == 1) , __FILE__, "Lbdybnzero=.false. only for" // &
        " fixed-boundary calculation with Lconstraint=-2")

    call fatal(Igeometry == 1 .and. rpol < vsmall, __FILE__, "poloidal extent of slab too small or negative ")

    call fatal(Igeometry == 1 .and. rtor < vsmall, __FILE__, "toroidal extent of slab too small or negative ")

    if (enforce_stell_sym) then
      Rbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
      Zbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
      Rws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
      Zwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
      Vnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
      Bnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    end if

    call fatal(abs(tflux(Nvol)) < vsmall, __FILE__, "enclosed toroidal flux cannot be 0.0_rkind ")

    toroidalflux = tflux(Nvol)! toroidal flux is a local variable: SRH:

    tflux(1:Mvol) = tflux(1:Mvol) / toroidalflux ! normalize toroidal flux
    pflux(1:Mvol) = pflux(1:Mvol) / toroidalflux ! normalize poloidal flux

    call fatal(tflux(1) < 0.0_rkind, __FILE__, "enclosed toroidal flux cannot be 0.0_rkind ")

    do vvol = 2, Mvol
      !FATAL( readin, tflux(vvol)-tflux(vvol-1).lt.small, toroidal flux is not monotonic )
    end do

    if ( (Lfreebound == 1) .and. (Lconstraint == 3) ) then

      Ivolume(Mvol) = Ivolume(Mvol - 1)! Ensure vacuum in vacuum region

      toroidalcurrent = Ivolume(Mvol) + sum(Isurf(1:Mvol - 1))

      if (abs(curtor) > vsmall) then
        call fatal(toroidalcurrent == 0, __FILE__, "Incompatible current profiles and toroidal linking current")

        Ivolume(1:Mvol) = Ivolume(1:Mvol) * curtor / toroidalcurrent
        Isurf(1:Mvol - 1) = Isurf(1:Mvol - 1) * curtor / toroidalcurrent

      else
        call fatal(abs(toroidalcurrent) > vsmall, __FILE__, "Incompatible current profiles and toroidal linking current")

        ! No rescaling if profiles have an overall zero toroidal current
      end if
    end if

    do vvol = 1, Mvol
      call fatal(Lrad(vvol) < 2, __FILE__, "require Chebyshev resolution Lrad > 2 so that Lagrange constraints can be  &
          &         satisfied ")

    end do

    if (Igeometry >= 2 .and. Lrad(1) < Mpol) then
      write(ounit, '("readin : ",f10.2," : Minimum Lrad(1) is Mpol, automatically adjusted it to Mpol+4")')cput - cpus
      Lrad(1) = Mpol + 4
    end if
    call fatal(mupfits <= 0, __FILE__, "must give ma01aa:hybrj a postive integer value for the maximum iterations =    &
        &     mupfits given on input ")

    write(ounit, '("readin : ", 10x ," : ")')

    write(ounit, 1020)cput - cpus, Linitialize, LautoinitBn, Lzerovac, Ndiscrete
    write(ounit, 1021)Nquad, iMpol, iNtor

    1020 format("readin : ", f10.2, " : Linitialize=", i3, " ;LautoinitBn=", i3, " ; Lzerovac=", i2, " ; Ndiscrete=", i2, " ")
    1021 format("readin : ", 10x, " : Nquad=", i4, " ; iMpol=", i4, " ; iNtor=", i4, " ")

    call fatal(Ndiscrete <= 0, __FILE__, "error ")

    write(ounit, '("readin : ", 10x ," : ")')

    write(ounit, '("readin : ", 10x ," : ")')

    write(ounit, 1041)escale, opsilon, pcondense, epsilon, wpoloidal, upsilon
    write(ounit, 1043)mfreeits, gBntol, gBnbld
    write(ounit, 1044)vcasingeps, vcasingtol, vcasingits, vcasingper, vcNt, vcNz

    1041 format("readin : ", 10x, " : escale=", es13.5, " ; opsilon=", es13.5, " ; pcondense=", f7.3, " ; epsilon=", es13.5, " ; wpoloidal &
        &       =", f7.4, " ; upsilon=", es13.5, " ")
    1043 format("readin : ", 10x, " : mfreeits=", i4, " ; gBntol=", es13.5, " ; gBnbld=", es13.5, " ")
    1044 format("readin : ", 10x, " : vcasingeps=", es13.5, " ; vcasingtol=", es13.5, " ; vcasingits=", i6, " ; vcasingper=", i6, " ; vcNt=", i6, " ; vcNz=", i6, " ")

    call fatal(escale < 0.0_rkind, __FILE__, "error ")

    call fatal(pcondense < 1.0_rkind, __FILE__, "error ")

    call fatal(Igeometry == 3 .and. pcondense <= 0.0_rkind, __FILE__, "pcondense must be positive ")

    write(ounit, '("readin : ", 10x ," : ")')

    write(ounit, 1050)cput - cpus, odetol, nPpts

    1050 format("readin : ", f10.2, " : odetol=", es10.2, " ; nPpts=", i6, " ")

    call fatal(odetol <= 0.0_rkind, __FILE__, "input error ")

    write(ounit, '("readin : ", 10x ," : ")')
  end subroutine check_inputs

  subroutine broadcast_inputs
    real(rkind) :: cput

    call broadcast(ext, 100, 0)

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : broadcasting physicslist from ext.sp")')cput - cpus
    end if

    call broadcast(Igeometry, 0)
    call broadcast(enforce_stell_sym, 0)
    call broadcast(Lfreebound, 0)
    call broadcast(Lbnszero, 0)
    call broadcast(phiedge, 0)
    call broadcast(curtor, 0)
    call broadcast(curpol, 0)
    call broadcast(gamma, 0)
    call broadcast(Nfp, 0)
    call broadcast(Nvol, 0)
    call broadcast(Mpol, 0)
    call broadcast(Ntor, 0)
    call broadcast(Lrad, MNvol + 1, 0)
    call broadcast(tflux, MNvol + 1, 0)
    call broadcast(pflux, MNvol + 1, 0)
    call broadcast(helicity, MNvol, 0)
    call broadcast(pscale, 0)
    call broadcast(pressure, MNvol + 1, 0)
    call broadcast(Ladiabatic, 0)
    call broadcast(adiabatic, MNvol + 1, 0)
    call broadcast(mu, MNvol + 1, 0)
    call broadcast(Ivolume, MNvol + 1, 0)
    call broadcast(Isurf, MNvol + 1, 0)
    call broadcast(Lconstraint, 0)
    call broadcast(pl, MNvol, 0)
    call broadcast(ql, MNvol, 0)
    call broadcast(pr, MNvol, 0)
    call broadcast(qr, MNvol, 0)
    call broadcast(iota, MNvol, 0)
    call broadcast(lp, MNvol, 0)
    call broadcast(lq, MNvol, 0)
    call broadcast(rp, MNvol, 0)
    call broadcast(rq, MNvol, 0)
    call broadcast(oita, MNvol, 0)
    call broadcast(mupftol, 0)
    call broadcast(mupfits, 0)
    call broadcast(Lreflect, 0)
    call broadcast(rpol, 0)
    call broadcast(rtor, 0)

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : broadcasting numericlist from ext.sp")')cput - cpus
    end if

    call broadcast(Linitialize, 0)
    call broadcast(LautoinitBn, 0)
    call broadcast(Lzerovac, 0)
    call broadcast(Ndiscrete, 0)
    call broadcast(Nquad, 0)
    call broadcast(iMpol, 0)
    call broadcast(iNtor, 0)
    call broadcast(Lsparse, 0)
    call broadcast(Lsvdiota, 0)
    call broadcast(Lrzaxis, 0)
    call broadcast(Ntoraxis, 0)
    call broadcast(Lvcgrid, 0)

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : broadcasting globallist from ext.sp")')cput - cpus
    end if

    call broadcast(escale, 0)
    call broadcast(opsilon, 0)
    call broadcast(pcondense, 0)
    call broadcast(epsilon, 0)
    call broadcast(wpoloidal, 0)
    call broadcast(upsilon, 0)
    call broadcast(mfreeits, 0)
    call broadcast(gBntol, 0)
    call broadcast(gBnbld, 0)
    call broadcast(vcasingeps, 0)
    call broadcast(vcasingtol, 0)
    call broadcast(vcasingits, 0)
    call broadcast(vcasingper, 0)
    call broadcast(vcNt, 0)
    call broadcast(vcNz, 0)

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : broadcasting diagnosticslist from ext.sp")')cput - cpus
    end if

    call broadcast(odetol, 0)
    call broadcast(nPpts, 0)
    call broadcast(Ppts, 0)
    call broadcast(nPtrj, MNvol + 1, 0)
    call broadcast(LHevalues, 0)
    call broadcast(LHevectors, 0)
    call broadcast(Ltransform, 0)
    call broadcast(LHmatrix, 0)
    call broadcast(Lerrortype, 0)
    call broadcast(Ngrid, 0)
    call broadcast(Lcheck, 0)

    if (Wreadin) then
      cput = MPI_Wtime()
      write(ounit, '("readin : ",f10.2," : broadcasting screenlist from ext.sp")')cput - cpus
    end if

    call broadcast(Wreadin, 0)
    call broadcast(Wwrtend, 0)
    call broadcast(Wmacros, 0)
  end subroutine broadcast_inputs

  subroutine wrtend
    real(rkind) :: cput

    integer :: vvol !< iteration variable over all nested volumes
    integer :: imn !< iteration variable for all Fourier harmonics
    integer :: ii !< iteration variable for all Fourier harmonics
    integer :: mm !< current poloidal mode number
    integer :: nn !< current toroidal mode number

    if (myid == 0) then

      open(iunit, file=trim(ext) // ".sp.end", status="unknown")! restart input file

      write(iunit, '("&physicslist")')
      write(iunit, '(" Igeometry   = ",i9        )')Igeometry
      write(iunit, '(" enforce_stell_sym   = ",L2        )')enforce_stell_sym
      write(iunit, '(" Lfreebound  = ",i9        )')Lfreebound
      write(iunit, '(" Lbnszero    = ",L2        )')Lbnszero
      write(iunit, '(" phiedge     = ",es23.15   )')phiedge
      write(iunit, '(" curtor      = ",es23.15   )')curtor
      write(iunit, '(" curpol      = ",es23.15   )')curpol
      write(iunit, '(" gamma       = ",es23.15   )')gamma
      write(iunit, '(" Nfp         = ",i9        )')Nfp
      write(iunit, '(" Nvol        = ",i9        )')Nvol
      write(iunit, '(" Mpol        = ",i9        )')Mpol
      write(iunit, '(" Ntor        = ",i9        )')Ntor
      write(iunit, '(" Lrad        = ",257i23    )')Lrad(1:Mvol)
      write(iunit, '(" tflux       = ",257es23.15)')tflux(1:Mvol)
      write(iunit, '(" pflux       = ",257es23.15)')pflux(1:Mvol)
      write(iunit, '(" helicity    = ",256es23.15)')helicity(1:Mvol)
      write(iunit, '(" pscale      = ",es23.15   )')pscale
      write(iunit, '(" Ladiabatic  = ",i9        )')Ladiabatic
      write(iunit, '(" pressure    = ",257es23.15)')pressure(1:Mvol)
      write(iunit, '(" adiabatic   = ",257es23.15)')adiabatic(1:Mvol)
      write(iunit, '(" mu          = ",257es23.15)')mu(1:Mvol)
      write(iunit, '(" Ivolume     = ",257es23.15)')Ivolume(1:Mvol)
      write(iunit, '(" Isurf       = ",257es23.15)')IPDt(1:Mvol)! Prints the actual surf current, not the targeted
      ! 1.0_rkind
      write(iunit, '(" Lconstraint = ",i9        )')Lconstraint
      write(iunit, '(" pl          = ",257i23    )')pl(0:Mvol)
      write(iunit, '(" ql          = ",257i23    )')ql(0:Mvol)
      write(iunit, '(" pr          = ",257i23    )')pr(0:Mvol)
      write(iunit, '(" qr          = ",257i23    )')qr(0:Mvol)
      write(iunit, '(" iota        = ",257es23.15)')iota(0:Mvol)
      write(iunit, '(" lp          = ",257i23    )')lp(0:Mvol)
      write(iunit, '(" lq          = ",257i23    )')lq(0:Mvol)
      write(iunit, '(" rp          = ",257i23    )')rp(0:Mvol)
      write(iunit, '(" rq          = ",257i23    )')rq(0:Mvol)
      write(iunit, '(" oita        = ",257es23.15)')oita(0:Mvol)
      write(iunit, '(" mupftol     = ",es23.15   )')mupftol
      write(iunit, '(" mupfits     = ",i9        )')mupfits
      write(iunit, '(" Lreflect    = ",i9        )')Lreflect
      write(iunit, '(" rpol        = ",es23.15   )')rpol
      write(iunit, '(" rtor        = ",es23.15   )')rtor

      if (Lfreebound == 1 .or. Zbs(0, 1) > 0.0_rkind) then
        do ii = 1, mn
          mm = im(ii)
          nn = in(ii) / Nfp
          Rbc(nn, mm) = iRbc(ii, Nvol)
          Zbs(nn, mm) = iZbs(ii, Nvol)
          Vns(nn, mm) = iVns(ii)
          Bns(nn, mm) = iBns(ii)
          Rbs(nn, mm) = iRbs(ii, Nvol)
          Zbc(nn, mm) = iZbc(ii, Nvol)
          Vnc(nn, mm) = iVnc(ii)
          Bnc(nn, mm) = iBnc(ii)
          Rwc(nn, mm) = iRbc(ii, Mvol)
          Zws(nn, mm) = iZbs(ii, Mvol)
          Rws(nn, mm) = iRbs(ii, Mvol)
          Zwc(nn, mm) = iZbc(ii, Mvol)
        end do
      end if

      write(iunit, '(" Rac         = ",99es23.15)')iRbc(1:Ntor + 1, 0)
      write(iunit, '(" Zas         = ",99es23.15)')iZbs(1:Ntor + 1, 0)
      write(iunit, '(" Ras         = ",99es23.15)')iRbs(1:Ntor + 1, 0)
      write(iunit, '(" Zac         = ",99es23.15)')iZbc(1:Ntor + 1, 0)

      do mm = 0, Mpol ! will write out the plasma boundary harmonics
        do nn = -Ntor, Ntor

          if (mm == 0 .and. nn < 0) cycle ! these modes are always excluded

          select case (mm)
          case (0:9)
            if (nn < -9 .and. nn > -99) write(iunit, 1000)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1001)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1002)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1001)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
          case (10:99)
            if (nn < -9 .and. nn > -99) write(iunit, 1003)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1004)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1005)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1004)nn, mm, Rbc(nn, mm), nn, mm, Zbs(nn, mm), nn, mm, Rbs(nn,&
                & mm), nn, mm, Zbc(nn, mm)
          end select

        end do
      end do

      do mm = 0, Mpol ! will write out the computation domain harmonics: (only relevant in free-boundary case)
        do nn = -Ntor, Ntor

          if (mm == 0 .and. nn < 0) cycle ! these modes are always excluded

          select case (mm)
          case (0:9)
            if (nn < -9 .and. nn > -99) write(iunit, 1010)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1011)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1012)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1011)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
          case (10:99)
            if (nn < -9 .and. nn > -99) write(iunit, 1013)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1014)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1015)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1014)nn, mm, Rwc(nn, mm), nn, mm, Zws(nn, mm), nn, mm, Rws(nn,&
                & mm), nn, mm, Zwc(nn, mm)
          end select

        end do
      end do

      do mm = 0, Mpol ! will write out the computation domain harmonics: (only relevant in free-boundary case)
        do nn = -Ntor, Ntor

          if (mm == 0 .and. nn < 0) cycle ! these modes are always excluded

          select case (mm)
          case (0:9)
            if (nn < -9 .and. nn > -99) write(iunit, 1020)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1021)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1022)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1021)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
          case (10:99)
            if (nn < -9 .and. nn > -99) write(iunit, 1023)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn < 0 .and. nn >= -9) write(iunit, 1024)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn >= 0 .and. nn <= 9) write(iunit, 1025)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
            if (nn > 9 .and. nn <= 99) write(iunit, 1024)nn, mm, Vns(nn, mm), nn, mm, Bns(nn, mm), nn, mm, Vnc(nn,&
                & mm), nn, mm, Bnc(nn, mm)
          end select

        end do
      end do

      1000 format("Rbc(", i3, ",", i1, ")", 2x, "=", es23.15, " Zbs(", i3, ",", i1, ")", 2x, "=", es23.15, " Rbs(", i3, ",",&
          & i1, ")", 2x, "=", es23.15, " Zbc(", i3, ",", i1, ")", 2x, "=", es23.15)
      1001 format("Rbc(", i2, ",", i1, ")", 3x, "=", es23.15, " Zbs(", i2, ",", i1, ")", 3x, "=", es23.15, " Rbs(", i2, ",",&
          & i1, ")", 3x, "=", es23.15, " Zbc(", i2, ",", i1, ")", 3x, "=", es23.15)
      1002 format("Rbc(", i1, ",", i1, ")", 4x, "=", es23.15, " Zbs(", i1, ",", i1, ")", 4x, "=", es23.15, " Rbs(", i1, ",",&
          & i1, ")", 4x, "=", es23.15, " Zbc(", i1, ",", i1, ")", 4x, "=", es23.15)
      1003 format("Rbc(", i3, ",", i2, ")", 1x, "=", es23.15, " Zbs(", i3, ",", i2, ")", 1x, "=", es23.15, " Rbs(", i3, ",",&
          & i2, ")", 1x, "=", es23.15, " Zbc(", i3, ",", i2, ")", 1x, "=", es23.15)
      1004 format("Rbc(", i2, ",", i2, ")", 2x, "=", es23.15, " Zbs(", i2, ",", i2, ")", 2x, "=", es23.15, " Rbs(", i2, ",",&
          & i2, ")", 2x, "=", es23.15, " Zbc(", i2, ",", i2, ")", 2x, "=", es23.15)
      1005 format("Rbc(", i1, ",", i2, ")", 3x, "=", es23.15, " Zbs(", i1, ",", i2, ")", 3x, "=", es23.15, " Rbs(", i1, ",",&
          & i2, ")", 3x, "=", es23.15, " Zbc(", i1, ",", i2, ")", 3x, "=", es23.15)

      1010 format("Rwc(", i3, ",", i1, ")", 2x, "=", es23.15, " Zws(", i3, ",", i1, ")", 2x, "=", es23.15, " Rws(", i3, ",",&
          & i1, ")", 2x, "=", es23.15, " Zwc(", i3, ",", i1, ")", 2x, "=", es23.15)
      1011 format("Rwc(", i2, ",", i1, ")", 3x, "=", es23.15, " Zws(", i2, ",", i1, ")", 3x, "=", es23.15, " Rws(", i2, ",",&
          & i1, ")", 3x, "=", es23.15, " Zwc(", i2, ",", i1, ")", 3x, "=", es23.15)
      1012 format("Rwc(", i1, ",", i1, ")", 4x, "=", es23.15, " Zws(", i1, ",", i1, ")", 4x, "=", es23.15, " Rws(", i1, ",",&
          & i1, ")", 4x, "=", es23.15, " Zwc(", i1, ",", i1, ")", 4x, "=", es23.15)
      1013 format("Rwc(", i3, ",", i2, ")", 1x, "=", es23.15, " Zws(", i3, ",", i2, ")", 1x, "=", es23.15, " Rws(", i3, ",",&
          & i2, ")", 1x, "=", es23.15, " Zwc(", i3, ",", i2, ")", 1x, "=", es23.15)
      1014 format("Rwc(", i2, ",", i2, ")", 2x, "=", es23.15, " Zws(", i2, ",", i2, ")", 2x, "=", es23.15, " Rws(", i2, ",",&
          & i2, ")", 2x, "=", es23.15, " Zwc(", i2, ",", i2, ")", 2x, "=", es23.15)
      1015 format("Rwc(", i1, ",", i2, ")", 3x, "=", es23.15, " Zws(", i1, ",", i2, ")", 3x, "=", es23.15, " Rws(", i1, ",",&
          & i2, ")", 3x, "=", es23.15, " Zwc(", i1, ",", i2, ")", 3x, "=", es23.15)

      1020 format("Vns(", i3, ",", i1, ")", 2x, "=", es23.15, " Bns(", i3, ",", i1, ")", 2x, "=", es23.15, " Vnc(", i3, ",",&
          & i1, ")", 2x, "=", es23.15, " Bnc(", i3, ",", i1, ")", 2x, "=", es23.15)
      1021 format("Vns(", i2, ",", i1, ")", 3x, "=", es23.15, " Bns(", i2, ",", i1, ")", 3x, "=", es23.15, " Vnc(", i2, ",",&
          & i1, ")", 3x, "=", es23.15, " Bnc(", i2, ",", i1, ")", 3x, "=", es23.15)
      1022 format("Vns(", i1, ",", i1, ")", 4x, "=", es23.15, " Bns(", i1, ",", i1, ")", 4x, "=", es23.15, " Vnc(", i1, ",",&
          & i1, ")", 4x, "=", es23.15, " Bnc(", i1, ",", i1, ")", 4x, "=", es23.15)
      1023 format("Vns(", i3, ",", i2, ")", 1x, "=", es23.15, " Bns(", i3, ",", i2, ")", 1x, "=", es23.15, " Vnc(", i3, ",",&
          & i2, ")", 1x, "=", es23.15, " Bnc(", i3, ",", i2, ")", 1x, "=", es23.15)
      1024 format("Vns(", i2, ",", i2, ")", 2x, "=", es23.15, " Bns(", i2, ",", i2, ")", 2x, "=", es23.15, " Vnc(", i2, ",",&
          & i2, ")", 2x, "=", es23.15, " Bnc(", i2, ",", i2, ")", 2x, "=", es23.15)
      1025 format("Vns(", i1, ",", i2, ")", 3x, "=", es23.15, " Bns(", i1, ",", i2, ")", 3x, "=", es23.15, " Vnc(", i1, ",",&
          & i2, ")", 3x, "=", es23.15, " Bnc(", i1, ",", i2, ")", 3x, "=", es23.15)

      write(iunit, '("/")')

      if (Wwrtend) then
        cput = MPI_Wtime()
        write(ounit, '("wrtend : ",f10.2," : myid=",i3," writing numericlist")')cput - cpus, myid
      end if

      write(iunit, '("&numericlist")')
      write(iunit, '(" Linitialize = ",i9            )')Linitialize
      write(iunit, '(" LautoinitBn = ",i9            )')LautoinitBn
      write(iunit, '(" Lzerovac    = ",i9            )')Lzerovac
      write(iunit, '(" Ndiscrete   = ",i9            )')Ndiscrete
      write(iunit, '(" Nquad       = ",i9            )')Nquad
      write(iunit, '(" iMpol       = ",i9            )')iMpol
      write(iunit, '(" iNtor       = ",i9            )')iNtor
      write(iunit, '(" Lsparse     = ",i9            )')Lsparse
      write(iunit, '(" Lsvdiota    = ",i9            )')Lsvdiota
      write(iunit, '(" Lrzaxis     = ",i9            )')Lrzaxis
      write(iunit, '(" Ntoraxis    = ",i9            )')Ntoraxis
      write(iunit, '(" Lvcgrid     = ",i9            )')Lvcgrid
      write(iunit, '("/")')

      if (Wwrtend) then
        cput = MPI_Wtime()
        write(ounit, '("wrtend : ",f10.2," : myid=",i3," writing globallist")')cput - cpus, myid
      end if

      write(iunit, '("&globallist")')
      write(iunit, '(" escale      = ",es23.15       )')escale
      write(iunit, '(" opsilon     = ",es23.15       )')opsilon
      write(iunit, '(" pcondense   = ",es23.15       )')pcondense
      write(iunit, '(" epsilon     = ",es23.15       )')epsilon
      write(iunit, '(" wpoloidal   = ",es23.15       )')wpoloidal
      write(iunit, '(" upsilon     = ",es23.15       )')upsilon
      write(iunit, '(" mfreeits    = ",i9            )')mfreeits
      write(iunit, '(" gBntol      = ",es23.15       )')gBntol
      write(iunit, '(" gBnbld      = ",es23.15       )')gBnbld
      write(iunit, '(" vcasingeps  = ",es23.15       )')vcasingeps
      write(iunit, '(" vcasingtol  = ",es23.15       )')vcasingtol
      write(iunit, '(" vcasingits  = ",i9            )')vcasingits
      write(iunit, '(" vcasingper  = ",i9            )')vcasingper
      write(iunit, '(" vcNt        = ",i9            )')vcNt
      write(iunit, '(" vcNz        = ",i9            )')vcNz
      write(iunit, '("/")')

      if (Wwrtend) then
        cput = MPI_Wtime()
        write(ounit, '("wrtend : ",f10.2," : myid=",i3," writing diagnosticslist")')cput - cpus, myid
      end if

      write(iunit, '("&diagnosticslist")')
      write(iunit, '(" odetol      = ",es23.15       )')odetol
      write(iunit, '(" nPpts       = ",i9            )')nPpts
      write(iunit, '(" Ppts        = ",es23.15       )')Ppts
      write(iunit, '(" nPtrj       = ",256i6         )')nPtrj(1:Mvol)
      write(iunit, '(" LHevalues   = ",L9            )')LHevalues
      write(iunit, '(" LHevectors  = ",L9            )')LHevectors
      write(iunit, '(" LHmatrix    = ",L9            )')LHmatrix
      write(iunit, '(" Lcheck      = ",i9            )')Lcheck
      write(iunit, '("/")')

      if (Wwrtend) then
        cput = MPI_Wtime()
        write(ounit, '("wrtend : ",f10.2," : myid=",i3," writing screenlist")')cput - cpus, myid
      end if

      write(iunit, '("&screenlist")')
      if (Wreadin) write(iunit, '(" Wreadin = ",L1                )')Wreadin
      if (Wwrtend) write(iunit, '(" Wwrtend = ",L1                )')Wwrtend
      if (Wmacros) write(iunit, '(" Wmacros = ",L1                )')Wmacros
      write(iunit, '("/")')

      ! write initial guess of interface geometry
      do imn = 1, mn
        write(iunit, '(2i6,1024es23.15)')im(imn), in(imn) / Nfp, (iRbc(imn, vvol), iZbs(imn, vvol), iRbs(imn, vvol),&
            & iZbc(imn, vvol), vvol = 1, Nvol)
      end do

      close(iunit)
    end if
  end subroutine wrtend

  subroutine IsMyVolume(vvol)
    integer, intent(in) :: vvol

    IsMyVolumeValue = -1 ! Error value - Problem with vvol / id
    if (myid /= modulo(vvol - 1, ncpu)) then
      IsMyVolumeValue = 0
    else
      IsMyVolumeValue = 1
    end if

  end subroutine IsMyVolume

  pure function WhichCpuID(vvol, ncpu)result(cpu_id)
    integer, intent(in) :: vvol
    integer, intent(in) :: ncpu
    integer :: cpu_id

    cpu_id = modulo(vvol - 1, ncpu)
  end function WhichCpuID
end module allglobal_mod
