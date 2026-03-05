module input_list_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: initialize_inputs

  ! The following parameters set the maximum allowed resolution:
  integer, parameter, public :: MNvol = 256 !< The maximum value of \c Nvol is \c MNvol=256.
  integer, parameter, public :: MMpol = 128 !< The maximum value of \c Mpol is \c MNpol=64.
  integer, parameter, public :: MNtor = 128 !< The maximum value of \c Ntor is \c MNtor=64.
  integer, parameter, public :: MNMAX = 128  +1 + 128 * (2 * 128 + 1) !< max number of modes MNtor + 1 + MMpol * (2 * MNtor + 1)

  ! note that all variables in namelist need to be broadcasted in readin
  integer, public :: Igeometry = 3 !< selects Cartesian, cylindrical or toroidal geometry
  !< <ul>
  !< <li> \c Igeometry=1 : Cartesian: geometry determined by \f$R\f$: </li>
  !< <li> \c Igeometry=2 : cylindrical: geometry determined by \f$R\f$: </li>
  !< <li> \c Igeometry=3 : toroidal: geometry determined by \f$R\f$ *and* \f$Z\f$: </li>
  !< </ul>
  logical, public :: enforce_stell_sym = .true. !< stellarator symmetry is enforced if \c Istellsym is .true.
  integer, public :: Lfreebound = 0 !< compute vacuum field surrounding plasma

  logical, public :: Lbnszero = .false. !< allows to run freeboundary with no normal field from plasma on comp. boundary

  real(rkind), public :: phiedge = 1.0_rkind !< total enclosed toroidal magnetic flux
  real(rkind), public :: curtor = 0.0_rkind !< total enclosed (toroidal) plasma current
  real(rkind), public :: curpol = 0.0_rkind !< total enclosed (poloidal) linking current
  real(rkind), public :: gamma = 0.0_rkind !< adiabatic index: cannot set \f$|\gamma| = 1\f$
  integer, public :: Nfp = 1 !< field periodicity
  !< <ul>
  !< <li> all Fourier representations are of the form \f$\cos(m\theta-n N \zeta)\f$, \f$\sin(m\theta-n N \zeta)\f$,
  !where \f$N\equiv\f$\c Nfp </li>
  !< <li> constraint: \c Nfp >= 1 </li>
  !< </ul>
  integer, public :: Nvol = 1 !< number of volumes
  !< <ul>
  !< <li> each volume \f${\cal V}_l\f$ is bounded by the \f${\cal I}_{l-1}\f$ and \f${\cal I}_{l}\f$ interfaces </li>
  !< <li> note that in cylindrical or toroidal geometry, \f${\cal I}_{0}\f$ is the degenerate coordinate axis </li>
  !< <li> constraint: \c Nvol<=MNvol </li>
  !< </ul>
  integer, public :: Mpol = 0 !< number of poloidal Fourier harmonics
  !< <ul>
  !< <li> all Fourier representations of doubly-periodic functions are of the form
  !< \f{eqnarray}{ f(\theta,\zeta) & = & \sum_{n=0}^{\texttt{Ntor}} f_{0,n}\cos(-n \, \texttt{Nfp} \, \zeta)
  !<                                   + \sum_{m=1}^{\texttt{Mpol}}\sum_{n=\texttt{-Ntor}}^{\texttt{Ntor}}
  !f_{m,n}\cos(m\theta-n \, \texttt{Nfp} \, \zeta),
  !< \f}
  !< Internally these "double" summations are written as a "single" summation,
  !< e.g. \f$f(\theta,\zeta) = \sum_j f_j \cos(m_j\theta-n_j\zeta)\f$. </li>
  !< </ul>
  integer, public :: Ntor = 0 !< number of toroidal Fourier harmonics
  !< <ul>
  !<                                   + \sum_{m=1}^{\texttt{Mpol}}\sum_{n=\texttt{-Ntor}}^{\texttt{Ntor}}
  !f_{m,n}\cos(m\theta-n \, \texttt{Nfp} \, \zeta),
  !< \f}
  !< Internally these "double" summations are written as a "single" summation,
  !< e.g. \f$f(\theta,\zeta) = \sum_j f_j \cos(m_j\theta-n_j\zeta)\f$. </li>
  !< </ul>
  integer, public :: Lrad(1:MNvol + 1) = 4 !< Chebyshev resolution in each volume
  !< <ul>
  !< <li> constraint : \c Lrad(1:Mvol) >= 2 </li>
  !< </ul>
  integer, public :: Lconstraint = -1 !< selects constraints: primarily used in construct_beltrami_field() and solve_beltrami_system().
  !< <ul>
  !< <li> if \c Lconstraint==-1, then in the plasma regions \f$\Delta\psi_t\f$, \f$\mu\f$ and \f$\Delta \psi_p\f$ are
  !*not* varied
  !<      and in the vacuum region (only for free-boundary) \f$\Delta\psi_t\f$ and \f$\Delta \psi_p\f$ are *not*
  !varied, and \f$\mu = 0\f$. </li>
  !< <li> if \c Lconstraint==0, then in the plasma regions \f$\Delta\psi_t\f$, \f$\mu\f$ and \f$\Delta \psi_p\f$ are
  !*not* varied
  !<      and in the vacuum region (only for free-boundary) \f$\Delta\psi_t\f$ and \f$\Delta \psi_p\f$ are varied to
  !match the
  !<      prescribed plasma current, \c curtor, and the "linking" current, \c curpol, and \f$\mu = 0\f$ </li>
  !< <li> if \c Lconstraint==1, then in the plasma regions \f$\mu\f$ and \f$\Delta\psi_p\f$ are adjusted
  !<      in order to satisfy the inner and outer interface transform constraints
  !<      (except in the simple torus, where the enclosed poloidal flux is irrelevant,
  !<      and only \f$\mu\f$ is varied to satisfy the outer interface transform constraint)
  !<      and in the vacuum region \f$\Delta\psi_t\f$ and \f$\Delta \psi_p\f$ are varied to match the transform
  !constraint on the boundary
  !<      and to obtain the prescribed linking current, \c curpol, and \f$\mu = 0\f$. </li>
  !< <li> \todo if \c Lconstraint==2, under reconstruction.
  !<
  !< </li>
  !< <li> if \c Lconstraint.eq.3 , then the \f$\mu\f$ and \f$\psi_p\f$ variables are adjusted
  !<      in order to satisfy the volume and surface toroidal current computed with lbpol()
  !<      (excepted in the inner most volume, where the volume current is irrelevant).
  !<      Not implemented yet in free boundary.</li>
  !< </ul>
  real(rkind), public :: tflux(1:MNvol + 1) = 0.0_rkind !< toroidal flux, \f$\psi_t\f$, enclosed by each interface
  !< <ul>
  !< <li> For each of the plasma volumes, this is a constraint: \c tflux is *not* varied </li>
  !< <li> For the vacuum region (only if \c Lfreebound==1), \c tflux  may be allowed to vary to match constraints </li>
  !< <li> Note that \c tflux  will be normalized so that \c tflux(Nvol) = 1.0,
  !<      so that \c tflux  is arbitrary up to a scale factor </li>
  !< <li> \sa phiedge </li>
  !< </ul>
  real(rkind), public :: pflux(1:MNvol + 1) = 0.0_rkind !< poloidal flux, \f$\psi_p\f$, enclosed by each interface
  real(rkind), public :: helicity(1:MNvol) = 0.0_rkind !< helicity, \f${\cal K}\f$, in each volume, \f${\cal V}_i\f$
  !< <ul>
  !< <li> on exit, \c helicity  is set to the computed values of \f${\cal K} \equiv \int {\bf A}\cdot{\bf B}\:dv\f$
  !</li>
  !< </ul>
  real(rkind), public :: pscale = 0.0_rkind !< pressure scale factor
  !< <ul>
  !< <li> the initial pressure profile is given by \c pscale  \f$*\f$ \c pressure </li>
  !< </ul>
  real(rkind), public :: pressure(1:MNvol + 1) = 0.0_rkind !< pressure in each volume
  !< <ul>
  !< <li> The pressure is *not* held constant, but \f$p_l V_l^\gamma = P_l\f$ *is* held constant,
  !<      where \f$P_l\f$ is determined by the initial pressures and the initial volumes, \f$V_l\f$. </li>
  !< <li> Note that if \c gamma==0.0, then \f$p_l \equiv P_l\f$. </li>
  !< <li> On output, the pressure is given by \f$p_l = P_l/V_l^\gamma\f$, where \f$V_l\f$ is the final volume. </li>
  !< <li> \c pressure is only used in calculation of interface force-balance. </li>
  !< </ul>
  integer, public :: Ladiabatic = 0 !< logical flag
  !< <ul>
  !< <li> If \c Ladiabatic==0, the adiabatic constants are determined by the initial pressure and volume. </li>
  !< <li> If \c Ladiabatic==1, the adiabatic constants are determined by the given input \c adiabatic. </li>
  !< </ul>
  real(rkind), public :: adiabatic(1:MNvol + 1) = 0.0_rkind !< adiabatic constants in each volume
  !< <ul>
  !< <li> The pressure is *not* held constant, but \f$p_l V_l^\gamma = P_l \equiv\f$\c adiabatic is constant. </li>
  !< <li> Note that if \c gamma==0.0, then \c pressure==adiabatic. </li>
  !< <li> \c pressure is only used in calculation of interface force-balance. </li>
  !< </ul>
  real(rkind), public :: mu(1:MNvol + 1) = 0.0_rkind !< helicity-multiplier, \f$\mu\f$, in each volume
  real(rkind), public :: Ivolume(1:MNvol + 1) = 0.0_rkind !< Toroidal current constraint normalized by \f$\mu_0\f$
  !(\f$I_{volume} = \mu_0\cdot [A]\f$), in each volume.
  !< This is a cumulative quantity: \f$I_{\mathcal{V},i} = \int_0^{\psi_{t,i}} \mathbf{J}\cdot\mathbf{dS}\f$.
  !< Physically, it represents the sum of all non-pressure driven currents.
  real(rkind), public :: Isurf(1:MNvol) = 0.0_rkind !< Toroidal current normalized by \f$\mu_0\f$ at each interface
  !(cumulative). This is the sum of all pressure driven currents.
  integer, public :: pl(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
  integer, public :: ql(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
  integer, public :: pr(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
  integer, public :: qr(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
  real(rkind), public :: iota(0:MNvol) = 0.0_rkind !< rotational-transform, \f$\mbox{$\,\iota\!\!$-}\f$, on inner side of each
  !interface
  !< <ul>
  !< <li> only relevant if illogical input for \c ql and \c qr are provided </li>
  !< </ul>
  integer, public :: lp(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
  integer, public :: lq(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
  integer, public :: rp(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
  integer, public :: rq(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
  !p_r)/(q_l+\gamma q_r)\f$,
  !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
  !<
  !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
  real(rkind), public :: oita(0:MNvol) = 0.0_rkind !< rotational-transform, \f$\mbox{$\,\iota\!\!$-}\f$, on outer side of each
  !interface
  !< <ul>
  !< <li> only relevant if illogical input for \c ql and \c qr are provided </li>
  !< </ul>
  real(rkind), public :: mupftol = 1.0e-14_rkind !< accuracy to which \f$\mu\f$ and \f$\Delta\psi_p\f$ are required
  !< <ul>
  !< <li> only relevant if constraints on transform, enclosed currents etc. are to be satisfied iteratively, see \c
  !Lconstraint </li>
  !< </ul>
  integer, public :: mupfits = 8 !< an upper limit on the transform/helicity constraint iterations
  !< <ul>
  !< <li> only relevant if constraints on transform, enclosed currents etc. are to be satisfied iteratively, see \c
  !Lconstraint </li>
  !< <li> constraint: \c mupfits > 0 </li>
  !< </ul>
  real(rkind), public :: rpol = 1.0_rkind !< poloidal extent of slab (effective radius)
  !< <ul>
  !< <li> only relevant if \c Igeometry==1 </li>
  !< <li> poloidal size is \f$L = 2\pi*\f$\c rpol </li>
  !< </ul>
  real(rkind), public :: rtor = 1.0_rkind !< toroidal extent of slab (effective radius)
  !< <ul>
  !< <li> only relevant if \c Igeometry==1 </li>
  !< <li> toroidal size is \f$L = 2\pi*\f$\c rtor </li>
  !< </ul>
  integer, public :: Lreflect = 0 !< =1 reflect the upper and lower bound in slab, =0 do not reflect

  logical, public :: Lbdybnzero = .true. !< for fixed-boundary (currently only for Lconstraint=-2)
  ! =.true. assume Bsups=0 on boundary
  ! =.false. obtain Bsups from Vns and Vnc input

  real(rkind), public :: Rac(0:MNtor) = 0.0_rkind !<     stellarator symmetric coordinate axis
  real(rkind), public :: Zas(0:MNtor) = 0.0_rkind !<     stellarator symmetric coordinate axis
  real(rkind), public :: Ras(0:MNtor) = 0.0_rkind !< non-stellarator symmetric coordinate axis
  real(rkind), public :: Zac(0:MNtor) = 0.0_rkind !< non-stellarator symmetric coordinate axis

  real(rkind), public :: Rbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components
  real(rkind), public :: Zbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components
  real(rkind), public :: Rbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components
  real(rkind), public :: Zbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components

  real(rkind), public :: Rwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components of wall
  real(rkind), public :: Zws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components of wall
  real(rkind), public :: Rws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components of wall
  real(rkind), public :: Zwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components of wall

  real(rkind), public :: Vns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric normal field at boundary:
  !vacuum component
  real(rkind), public :: Bns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric normal field at boundary:
  !plasma component
  real(rkind), public :: Vnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric normal field at boundary:
  !vacuum component
  real(rkind), public :: Bnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric normal field at boundary:
  !plasma component

  integer, allocatable, public :: mmRZRZ(:)  !< poloidal mode index helper array for interface guesses
  integer, allocatable, public :: nnRZRZ(:) !< toroidal mode index helper array for interface guesses
  real(rkind), allocatable, public :: allRZRZ(:, :, :) !< interface fourier harmonic array containing initial guesses for interfaces of the shape (4, 1:Nvol-1, 1:MNMAX)
  ! array used for reading interface Fourier harmonics from file
  integer, public :: num_modes

  integer, public :: Linitialize = 0 !< Used to initialize geometry using a regularization / extrapolation method
  !< <ul>
  !< <li> if \c Linitialize = \f$-I\f$ , where \f$I\f$ is a positive integer,
  !<      the geometry of the \f$i=1,N_V-I\f$ surfaces constructed by an extrapolation </li>
  !< <li> if \c Linitialize = 0, the geometry of the interior surfaces is provided after the namelists in the input
  !file </li>
  !< <li> if \c Linitialize = 1, the interior surfaces will be intialized as \f$R_{l,m,n} = R_{N,m,n}
  !\psi_{t,l}^{m/2}\f$,
  !<      where \f$R_{N,m,n}\f$ is the plasma boundary and \f$\psi_{t,l}\f$ is the given toroidal flux enclosed by the
  !<      \f$l\f$-th interface, normalized to the total enclosed toroidal flux
  !<      a similar extrapolation is used for \f$Z_{l,m,n}\f$ </li>
  !< <li> Note that the Fourier harmonics of the boundary is *always* given by the \c Rbc and \c Zbs
  !<      given in \c physicslist. </li>
  !< <li> if \c Linitialize = 2, the interior surfaces *and the plasma boundary* will be intialized
  !<       as \f$R_{l,m,n} = R_{W,m,n} \psi_{t,l}^{m/2}\f$, where \f$R_{W,m,n}\f$ is the computational boundary
  !<       and \f$\psi_{t,l}\f$ is the given toroidal flux enclosed by the \f$l\f$-th interface, normalized to the
  !total enclosed toroidal flux
  !<       a similar extrapolation is used for \f$Z_{l,m,n}\f$ </li>
  !< <li> Note that, for free-boundary calculations, the Fourier harmonics of the computational boundary
  !<      are *always* given by the \c Rwc and \c Zws given in \c physicslist. </li>
  !< <li> if \c Linitialize = 1, 2, it is not required to provide the geometry of the interfaces after the namelists
  !</li>
  !< </ul>
  integer, public :: LautoinitBn = 1 !< Used to initialize \f$B_{ns}\f$ using an initial fixed-boundary calculation
  !< <ul>
  !< <li> only relevant if \c Lfreebound = 1 </li>
  !< <li> user-supplied \c Bns will only be considered if \c LautoinitBn = 0 </li>
  !< </ul>
  integer, public :: Lzerovac = 0 !< Used to adjust vacuum field to cancel plasma field on computational boundary
  !< <ul>
  !< <li> only relevant if \c Lfreebound = 1 </li>
  !< </ul>
  integer, public :: Ndiscrete = 2 !< resolution of the real space grid on which fast Fourier transforms are performed is given
  !by \c Ndiscrete*Mpol*4
  !< <ul>
  !< <li> constraint \c Ndiscrete>0 </li>
  !< </ul>
  integer, public :: Nquad = -1 !< Resolution of the Gaussian quadrature
  !< <ul>
  !< <li> The resolution of the Gaussian quadrature, \f$\displaystyle \int \!\! f(s) ds = \sum_k \omega_k f(s_k)\f$,
  !<      in each volume is given by \c Iquad\f$_v\f$,  </li>
  !< <li> \c Iquad\f$_v\f$ is set in initialize_internal_arrays() </li>
  !< </ul>
  integer, public :: iMpol = -4 !< Fourier resolution of straight-fieldline angle on interfaces
  !< <ul>
  !< <li> the rotational-transform on the interfaces is determined by a transformation to the straight-fieldline angle,
  !<      with poloidal resolution given by \c iMpol </li>
  !< <li> if \c iMpol<=0, then \c iMpol = Mpol - iMpol </li>
  !< </ul>
  integer, public :: iNtor = -4 !< Fourier resolution of straight-fieldline angle on interfaces
  !< <ul>
  !< <li> the rotational-transform on the interfaces is determined by a transformation to the straight-fieldline angle,
  !<      with toroidal resolution given by \c iNtor </li>
  !< <li> if \c iNtor<=0 then \c iNtor = Ntor - iNtor </li>
  !< <li> if \c Ntor==0, then the toroidal resolution of the angle transformation is set \c lNtor = 0 </li>
  !< </ul>
  integer, public :: Lsparse = 0 !< controls method used to solve for rotational-transform on interfaces
  !< <ul>
  !< <li> if \c Lsparse = 0, the transformation to the straight-fieldline angle is computed in Fourier space
  !<      using a dense matrix solver,  \c F04AAF </li>
  !< <li> if \c Lsparse = 1, the transformation to the straight-fieldline angle is computed in real space
  !<      using a dense matrix solver,  \c F04ATF </li>
  !< <li> if \c Lsparse = 2, the transformation to the straight-fieldline angle is computed in real space
  !<      using a sparse matrix solver, \c F11DEF </li>
  !< <li> if \c Lsparse = 3, the different methods for constructing the straight-fieldline angle are compared </li>
  !< </ul>
  integer, public :: Lsvdiota = 0 !< controls method used to solve for rotational-transform on interfaces
  !< only relevant if \c Lsparse = 0
  !< <ul>
  !< <li> if \c Lsvdiota = 0, use standard linear solver to construct straight fieldline angle transformation </li>
  !< <li> if \c Lsvdiota = 1, use SVD method to compute rotational-transform </li>
  !< </ul>
  integer, public :: imethod = 3 !< controls iterative solution to sparse matrix
  !< arising in real-space transformation to the straight-fieldline angle
  !< only relevant if \c Lsparse.eq.2: \see compute_rotational_transform() for details
  !< <ul>
  !< <li> if \c imethod = 1, the method is \c RGMRES   </li>
  !< <li> if \c imethod = 2, the method is \c CGS      </li>
  !< <li> if \c imethod = 3, the method is \c BICGSTAB </li>
  !< </ul>
  integer, public :: iorder = 2 !< controls real-space grid resolution for constructing the straight-fieldline angle
  !< only relevant if \c Lsparse>0
  !<
  !< determines order of finite-difference approximation to the derivatives
  !< <ul>
  !< <li> if \c iorder = 2,  </li>
  !< <li> if \c iorder = 4,  </li>
  !< <li> if \c iorder = 6,  </li>
  !< </ul>
  integer, public :: iprecon = 0 !< controls iterative solution to sparse matrix arising in real-space transformation
  !< to the straight-fieldline angle
  !< only relevant if \c Lsparse.eq.2: \see compute_rotational_transform() for details
  !< <ul>
  !< <li> if \c iprecon = 0, the preconditioner is `N' </li>
  !< <li> if \c iprecon = 1, the preconditioner is `J' </li>
  !< <li> if \c iprecon = 2, the preconditioner is `S' </li>
  !< </ul>
  real(rkind), public :: iotatol = -1.0_rkind !< tolerance required for iterative construction of straight-fieldline angle
  !< only relevant if \c Lsparse.ge.2
  integer, public :: Mregular = -1 !< maximum regularization factor
  !< <ul>
  !< <li> if \c Mregular.ge.2, then \c regumm \f$_i\f$ = \c Mregular \f$/ 2 \f$ where \c m \f$_i > \f$ \c Mregular </li>
  !< </ul>
  integer, public :: Lrzaxis = 1 !< controls the guess of geometry axis in the innermost volume or initialization of interfaces
  !< <ul>
  !< <li> if \c iprecon = 1, the centroid is used </li>
  !< <li> if \c iprecon = 2, the Jacobian \f$m=1\f$ harmonic elimination method is used </li>
  !< </ul>
  integer, public :: Ntoraxis = 3 !< the number of \f$n\f$ harmonics used in the Jacobian \f$m=1\f$ harmonic elimination method
  !< only relevant if \c Lrzaxis.ge.1 .

  integer, public :: Lvcgrid = 1 !< Which method to use for the virtual casing integral.
  !< 0 = adaptive integration routine with guaranteed accuracy
  !< 1 = fixed resolution grid

  integer, public :: Linitgues = 1 !< controls how initial guess for Beltrami field is constructed
  !< <ul>
  !< <li> only relevant for routines that require an initial guess for the Beltrami fields, such as the SQP and Newton
  !methods,
  !<      or the sparse linear solver
  !< <li> if \c Linitgues = 0, the initial guess for the Beltrami field is trivial
  !< <li> if \c Linitgues = 1, the initial guess for the Beltrami field is an integrable approximation
  !< <li> if \c Linitgues = 2, the initial guess for the Beltrami field is read from file
  !< <li> if \c Linitgues = 3, the initial guess for the Beltrami field will be randomized with the maximum \c
  !maxrndgues
  !< </ul>
  integer, public :: Lposdef = 0 !< redundant
  real(rkind), public :: maxrndgues = 1.0_rkind !< the maximum random number of the Beltrami field if \c Linitgues = 3

  real(rkind), public :: escale = 0.0_rkind !< controls the weight factor, \c BBweight, in the force-imbalance harmonics
  !< <ul>
  !< <li> \c BBweight(i) \f$\displaystyle \equiv \texttt{opsilon} \times \exp\left[-\texttt{escale} \times
  !(m_i^2+n_i^2) \right]\f$ </li>
  !< <li> defined in initialize_internal_arrays() : used in dforce() </li>
  !< <li> also see Eqn.\f$(\ref{eq:forcebalancemn_global})\f$ </li>
  !< </ul>
  real(rkind), public :: opsilon = 1.0_rkind !< weighting of force-imbalance
  !< <ul>
  !< <li> used in dforce(): also see Eqn.\f$(\ref{eq:forcebalancemn_global})\f$ </li>
  !< </ul>
  real(rkind), public :: pcondense = 2.0_rkind !< spectral condensation parameter
  !< <ul>
  !< <li> used in initialize_internal_arrays() to define \c mmpp(i) \f$\equiv m_i^p\f$, where \f$p\equiv \f$ \c pcondense </li>
  !< <li> the angle freedom is exploited to minimize \f$\displaystyle \texttt{epsilon} \sum_{i} m_i^p
  !(R_{i}^2+Z_{i}^2)\f$
  !<      with respect to tangential variations in the interface geometry </li>
  !< <li> also see Eqn.\f$(\ref{eq:spectralbalancemn_global})\f$ </li>
  !< </ul>
  real(rkind), public :: epsilon = 0.0_rkind !< weighting of spectral-width constraint
  !< <ul>
  !< <li> used in dforce(): also see Eqn.\f$(\ref{eq:spectralbalancemn_global})\f$ </li>
  !< </ul>
  real(rkind), public :: wpoloidal = 1.0_rkind !< "star-like" poloidal angle constraint radial exponential factor
  !< used in initialize_internal_arrays() to construct \c sweight
  real(rkind), public :: upsilon = 1.0_rkind !< weighting of "star-like" poloidal angle constraint
  !< used in initialize_internal_arrays() to construct \c sweight
  real(rkind), public :: forcetol = 1.0e-10_rkind !< required tolerance in force-balance error: only used as an initial check
  !< <ul>
  !< <li> if the initially supplied interfaces are consistent with force-balance to within \c forcetol
  !<      then the geometry of the interfaces is not altered </li>
  !< <li> if not, then the geometry of the interfaces is changed in order to bring the configuration into force balance
  !<      so that the geometry of interfaces is within \c c05xtol, defined below, of the true solution </li>
  !< <li> to force execution of either \c C05NDF or \c C05PDF, regardless of the initial force imbalance,
  !<      set \c forcetol < 0 </li>
  !< </ul>
  real(rkind), public :: c05xmax = 1.0e-06_rkind !< required tolerance in position, \f${\bf x} \equiv \{ R_{i,v}, Z_{i,v}\}\f$
  real(rkind), public :: c05xtol = 1.0e-12_rkind !< required tolerance in position, \f${\bf x} \equiv \{ R_{i,v}, Z_{i,v}\}\f$
  !< <ul>
  !< <li> used by both \c C05NDF and \c C05PDF: see the NAG documents for further details on how the error is defined
  !</li>
  !< <li> constraint \c c05xtol > 0.0 </li>
  !< </ul>
  real(rkind), public :: c05factor = 1.0e-02_rkind !< used to control initial step size in
  !<       \c C05NDF and \c C05PDF
  !< <ul>
  !< <li> constraint \c c05factor > 0.0 </li>
  !< </ul>
  logical, public :: LreadGF = .true. !< read \f$\nabla_{\bf x} {\bf F}\f$ from file \c ext.GF
  !< <ul>
  !< <li> only used in newton() </li>
  !< </ul>
  integer, public :: mfreeits = 0 !< maximum allowed free-boundary iterations
  !< <ul>
  !< <li> only used if \c Lfreebound = 1 </li>
  !< <li> only used in xspech() </li>
  !< </ul>
  real(rkind), public :: bnstol = 1.0e-06_rkind !< redundant
  real(rkind), public :: bnsblend = 0.666_rkind !< redundant
  real(rkind), public :: gBntol = 1.0e-06_rkind !< required tolerance in free-boundary iterations
  !< <ul>
  !< <li> only used if \c Lfreebound = 1 </li>
  !< <li> only used in xspech() </li>
  !< </ul>
  real(rkind), public :: gBnbld = 0.666_rkind !< normal blend
  !< <ul>
  !< <li> The "new" magnetic field at the computational boundary produced by the plasma currents is updated using a
  !Picard scheme:
  !<      \f{eqnarray}{ ({\bf B}\cdot{\bf n})^{j+1} =    \texttt{gBnbld}  \times ({\bf B}\cdot{\bf n})^{j}
  !<                                      + (1-\texttt{gBnbld}) \times ({\bf B}\cdot{\bf n})^{*},
  !<      \f}
  !<      where \f$j\f$ labels free-boundary iterations, and \f$({\bf B}\cdot{\bf n})^{*}\f$ is computed by virtual
  !casing. </li>
  !< <li> only used if \c Lfreebound = 1 </li>
  !< <li> only used in xspech() </li>
  !< </ul>
  real(rkind), public :: vcasingeps = 1.e-12_rkind !< regularization of Biot-Savart: see bnorml(), casing()
  real(rkind), public :: vcasingtol = 1.e-08_rkind !< accuracy on virtual casing integral: see bnorml(), casing()
  integer, public :: vcasingits = 8 !< minimum number of calls to adaptive virtual casing routine: see casing()
  integer, public :: vcasingper = 1 !< periods of integragion  in adaptive virtual casing routine: see casing()
  integer, public :: mcasingcal = 8 !< minimum number of calls to adaptive virtual casing routine: see casing(): redundant
  integer, public :: vcNt = 256 !< theta res. of the real space grid on which the surface current is computed in virtual casing [0, 2pi]
  integer, public :: vcNz = 256 !< zeta res. of the real space grid on which the surface current is computed in virtual casing [0, 2pi], over one field period

  real(rkind), public :: odetol = 1.0e-07_rkind !< o.d.e. integration tolerance for all field line tracing routines
  real(rkind), public :: absreq = 1.0e-08_rkind !< redundant
  real(rkind), public :: relreq = 1.0e-08_rkind !< redundant
  real(rkind), public :: absacc = 1.0e-04_rkind !< redundant
  real(rkind), public :: epsr = 1.0e-08_rkind !< redundant
  integer, public :: nPpts = 0 !< number of toroidal transits used (per trajectory) in following field lines
  !< for constructing Poincaré plots
  !< if \c nPpts<1, no Poincaré plot is constructed
  real(rkind), public :: Ppts = 0.0_rkind !< stands for Poincare plot theta start. Chose at which angle (normalized over
  !\f$\pi\f$) the Poincare field-line tracing start.
  integer, public :: nPtrj(1:MNvol + 1) = -1 !< number of trajectories in each annulus to be followed in constructing Poincaré
  !plot
  !< <ul>
  !< <li> if \c nPtrj(l)<0, then \c nPtrj(l) = Ni(l),
  !<       where \c Ni(l) is the grid resolution used to construct the Beltrami field in volume \f$l\f$ </li>
  !< </ul>
  logical, public :: LHevalues = .false. !< to compute eigenvalues of \f$\nabla {\bf F}\f$
  logical, public :: LHevectors = .false. !< to compute eigenvectors (and also eigenvalues) of \f$\nabla {\bf F}\f$
  logical, public :: LHmatrix = .false. !< to compute and write to file the elements of \f$\nabla {\bf F}\f$
  integer, public :: dpp = -1 !< perturbed harmonic
  integer, public :: dqq = -1 !< perturbed harmonic
  integer, public :: Lerrortype = 0 !< the type of error output for Lcheck=1
  integer, public :: Ngrid = -1 !< the number of points to output in the grid, -1 for Lrad(vvol)
  real(rkind), public :: dRZ = 1e-5_rkind !< difference in geometry for finite difference estimate (debug only)
  integer, public :: Lcheck = 0 !< implement various checks
  !< <ul>
  !< <li> if \c Lcheck = 0, no additional check on the calculation is performed </li>
  !< <li> if \c Lcheck = 1, the error in the current, i.e. \f$\nabla\times{\bf B}-\mu{\bf B}\f$ is computed as a
  !post-diagnostic </li>
  !< <li> if \c Lcheck = 2, the analytic derivatives of the interface transform w.r.t.
  !<      the helicity multiplier, \f$\mu\f$, and the enclosed poloidal flux, \f$\Delta\psi_p\f$, are compared to a
  !finite-difference estimate
  !<      <ul>
  !<      <li> only if \c Lconstraint==1 </li>
  !<      <li> only for \c dspec executable, i.e. must compile with \c DFLAGS = "-D DEBUG" </li>
  !<      </ul> </li>
  !< <li> if \c Lcheck = 3, the analytic derivatives of the volume w.r.t. interface Fourier harmonic
  !<      is compared to a finite-difference estimate
  !<      <ul>
  !<      <li> set \c forcetol sufficiently small and set \c LreadGF = F,
  !<           so that the matrix of second derivatives is calculated </li>
  !<      <li> only for \c dspec executable, i.e. must compile with \c DFLAGS = "-D DEBUG" </li>
  !<      </ul> </li>
  !< <li> if \c Lcheck = 4, the analytic calculation of the derivatives of the magnetic field, \f$B^2\f$, at the
  !interfaces
  !<      is compared to a finite-difference estimate
  !<      <ul>
  !<      <li> set \c forcetol sufficiently small </li>
  !<      <li> set \c LreadGF=F </li>
  !<      <li> only for \c dspec executable, i.e. must compile with \c DFLAGS = "-D DEBUG" </li>
  !<      </ul> </li>
  !< <li> if \c Lcheck = 5, the analytic calculation of the matrix of the derivatives of the force imbalance
  !<      is compared to a finite-difference estimate </li>
  !< <li> if \c Lcheck = 6, the virtual casing calculation is compared to \c xdiagno (Lazerson 2013 \cite
  !y2013_lazerson)
  !<      <ul>
  !<      <li> the input file for \c xdiagno is written by bnorml() </li>
  !<      <li> this provides the Cartesian coordinates on the computational boundary where the virtual casing routine
  !casing()
  !<           computes the magnetic field, with the values of the magnetic field being written to the screen for
  !comparison </li>
  !<      <li> must set \c Freebound=1, \c mfreeits!=0 </li>
  !<      <li> \c xdiagno must be executed manually </li>
  !<      </ul> </li>
  !< </ul>
  logical, public :: Ltiming = .false. !< to check timing
  logical, public :: Ltransform = .false. !< to evaluate iota and straight field line coordinates
  real(rkind), public :: fudge = 1.0e-00_rkind !< redundant
  real(rkind), public :: scaling = 1.0e-00_rkind !< redundant

  logical, public :: Wmanual = .false.
  logical, public :: Wrzaxis = .false.
  logical, public :: Wpackxi = .false.
  logical, public :: Wvolume = .false.
  logical, public :: Wcompute_coordinates = .false.
  logical, public :: Wbasefn = .false.
  logical, public :: Wmemory = .false.
  logical, public :: Wcompute_metric_quantities = .false.
  logical, public :: Wvolume_integrate_chebyshev = .false.
  logical, public :: Wmatrix = .false.
  logical, public :: Wspsint = .false.
  logical, public :: Wsolve_beltrami_system = .false.
  logical, public :: Wconstruct_beltrami_field = .false.
  logical, public :: Wpackab = .false.
  logical, public :: Wcompute_rotational_transform = .false.
  logical, public :: Wcompute_plasma_current = .false.
  logical, public :: Wdf00ab = .false.
  logical, public :: Wlforce = .false.
  logical, public :: Wintghs = .false.
  logical, public :: Wmtrxhs = .false.
  logical, public :: Wlbpol = .false.
  logical, public :: Wbroadcast_profile_data = .false.
  logical, public :: Wdfp100 = .false.
  logical, public :: Wdfp200 = .false.
  logical, public :: Wdforce = .false.
  logical, public :: Wnewton = .false.
  logical, public :: Wcasing = .false.
  logical, public :: Wbnorml = .false.
  logical, public :: Wcompute_beltrami_error = .false.
  logical, public :: Wconstruct_poincare_plot = .false.
  logical, public :: Wpp00ab = .false.
  logical, public :: Wget_position_vetor_x = .false.
  logical, public :: Whesian = .false.
  logical, public :: Wwrite_vector_potentials = .false.
  logical, public :: Wnumrec = .false.
  logical, public :: Wdcuhre = .false.
  logical, public :: Wminpack = .false.
  logical, public :: Wiqpack = .false.
  logical, public :: Wrksuite = .false.
  logical, public :: Wi1mach = .false.
  logical, public :: Wd1mach = .false.
  logical, public :: Wilut = .false.
  logical, public :: Witers = .false.
  logical, public :: Wsphdf5 = .false.
  logical, public :: Winitialize_internal_arrays = .false.
  logical, public :: Wglobal = .false.
  logical, public :: Wxspech = .false.
  logical, public :: Wbuild_vector_potential = .false. !< \todo: what is this?
  logical, public :: Wreadin = .false. !< write screen output of readin()
  logical, public :: Wwrtend = .false. !< write screen output of wrtend()
  logical, public :: Wmacros = .false. !< write screen output from expanded macros

  namelist /physicslist/ Igeometry, enforce_stell_sym, Lfreebound, Lbnszero, phiedge, curtor, curpol, gamma, Nfp, Nvol, Mpol, &
      Ntor, Lrad, Lconstraint, tflux, pflux, helicity, pscale, pressure, Ladiabatic, adiabatic, mu, Ivolume, Isurf, &
      pl, ql, pr, qr, iota, lp, lq, rp, rq, oita, mupftol, mupfits, rpol, rtor, Lreflect, Lbdybnzero, Rac, Zas, Ras, Zac, Rbc, &
      Zbs, Rbs, Zbc, Rwc, Zws, Rws, Zwc, Vns, Bns, Vnc, Bnc, allRZRZ, mmRZRZ, nnRZRZ
  public :: physicslist

  namelist /numericlist/ Linitialize, LautoinitBn, Lzerovac, Ndiscrete, Nquad, iMpol, iNtor, Lsparse, Lsvdiota, imethod, &
      iorder, iprecon, iotatol, Mregular, Lrzaxis, Ntoraxis, Lvcgrid
  public :: numericlist

  namelist /locallist/ Linitgues, maxrndgues, Lposdef
  public :: locallist

  namelist /globallist/ escale, opsilon, pcondense, epsilon, wpoloidal, upsilon, forcetol, c05xmax, c05xtol, &
      c05factor, LreadGF, mfreeits, bnstol, bnsblend, gBntol, gBnbld, vcasingeps, vcasingtol, vcasingits, vcasingper, &
      mcasingcal, vcNt, vcNz
  public :: globallist

  namelist /diagnosticslist/ odetol, absreq, relreq, absacc, epsr, nPpts, Ppts, nPtrj, LHevalues, LHevectors, LHmatrix, &
      dpp, dqq, Lerrortype, Ngrid, Lcheck, dRZ, Ltiming, Ltransform, fudge, scaling
  public :: diagnosticslist

  namelist /screenlist/ Wmanual, Wrzaxis, Wpackxi, Wvolume, Wcompute_coordinates, Wbasefn, Wmemory, Wcompute_metric_quantities, &
      Wvolume_integrate_chebyshev, Wmatrix, &
      Wspsint, Wsolve_beltrami_system, Wconstruct_beltrami_field, Wpackab, Wcompute_rotational_transform, Wcompute_plasma_current, &
      Wdf00ab, Wlforce, Wintghs, Wmtrxhs, Wlbpol, &
      Wbroadcast_profile_data, Wdfp100, Wdfp200, Wdforce, Wnewton, Wcasing, Wbnorml, Wcompute_beltrami_error, &
      Wconstruct_poincare_plot, Wpp00ab, &
      Wget_position_vetor_x, Whesian, Wwrite_vector_potentials, Wnumrec, Wdcuhre, Wminpack, Wiqpack, Wrksuite, Wi1mach, Wd1mach, &
      Wilut, Witers, Wsphdf5, &
      Winitialize_internal_arrays, Wglobal, Wxspech, Wbuild_vector_potential, Wreadin, Wwrtend, Wmacros
  public :: screenlist

contains

  ! The task of this subroutine is to initialize *all* input variables
  ! to a default known state. A subsequent call to readin() can be used
  ! to fill in the values from a text input file containing the
  ! appropriate input namelists. Alternatively, after calling this routine,
  ! the input variables can be modified from their default values in-memory,
  ! e.g., via the Python wrapper.
  !
  ! Note that since the variables in screenlist are auto-generated,
  ! they do not appear here and also are not touched by this routine.
  subroutine initialize_inputs()
    ! physicslist

    Igeometry = 3
    enforce_stell_sym = .true.
    Lfreebound = 0
    Lbnszero = .false.
    phiedge = 1.0_rkind
    curtor = 0.0_rkind
    curpol = 0.0_rkind
    gamma = 0.0_rkind
    Nfp = 1
    Nvol = 1
    Mpol = 0
    Ntor = 0
    Lrad(1:MNvol + 1) = 4
    Lconstraint = -1
    tflux(1:MNvol + 1) = 0.0_rkind
    pflux(1:MNvol + 1) = 0.0_rkind
    helicity(1:MNvol) = 0.0_rkind
    pscale = 0.0_rkind
    pressure(1:MNvol + 1) = 0.0_rkind
    Ladiabatic = 0
    adiabatic(1:MNvol + 1) = 0.0_rkind
    mu(1:MNvol + 1) = 0.0_rkind
    Ivolume(1:MNvol + 1) = 0.0_rkind
    Isurf(1:MNvol) = 0.0_rkind
    pl(0:MNvol) = 0
    ql(0:MNvol) = 0
    pr(0:MNvol) = 0
    qr(0:MNvol) = 0
    iota(0:MNvol) = 0.0_rkind
    lp(0:MNvol) = 0
    lq(0:MNvol) = 0
    rp(0:MNvol) = 0
    rq(0:MNvol) = 0
    oita(0:MNvol) = 0.0_rkind
    rpol = 1.0_rkind
    rtor = 1.0_rkind

    Rac(0:MNtor) = 0.0_rkind
    Zas(0:MNtor) = 0.0_rkind
    Ras(0:MNtor) = 0.0_rkind
    Zac(0:MNtor) = 0.0_rkind

    Rbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Zbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Rbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Zbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind

    Rwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Zws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Rws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Zwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind

    Vns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Bns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Vnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind
    Bnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind

    mupftol = 1.0e-14_rkind
    mupfits = 8

    Lreflect = 0

    Lbdybnzero = .true.

    num_modes = 0

    ! numericlist

    Linitialize = 0
    LautoinitBn = 1
    Lzerovac = 0
    Ndiscrete = 2
    Nquad = -1
    iMpol = -4
    iNtor = -4
    Lsparse = 0
    Lsvdiota = 0
    imethod = 3
    iorder = 2
    iprecon = 0
    iotatol = -1.0_rkind
    Mregular = -1
    Lrzaxis = 1
    Ntoraxis = 3
    Lvcgrid = 1

    ! locallist

    Linitgues = 1
    Lposdef = 0
    maxrndgues = 1.0_rkind

    ! globallist

    escale = 0.0_rkind
    opsilon = 1.0_rkind
    pcondense = 2.0_rkind
    epsilon = 0.0_rkind
    wpoloidal = 1.0_rkind
    upsilon = 1.0_rkind
    forcetol = 1.0e-10_rkind
    c05xmax = 1.0e-06_rkind
    c05xtol = 1.0e-12_rkind
    c05factor = 1.0e-02_rkind
    LreadGF = .true.
    mfreeits = 0
    bnstol = 1.0e-06_rkind
    bnsblend = 0.666_rkind
    gBntol = 1.0e-06_rkind
    gBnbld = 0.666_rkind
    vcasingeps = 1.e-12_rkind
    vcasingtol = 1.e-08_rkind
    vcasingits = 8
    vcasingper = 1
    mcasingcal = 8
    vcNt = 256
    vcNz = 256

    ! diagnosticslist

    odetol = 1.0e-07_rkind
    absreq = 1.0e-08_rkind
    relreq = 1.0e-08_rkind
    absacc = 1.0e-04_rkind
    epsr = 1.0e-08_rkind
    nPpts = 0
    Ppts = 0.0_rkind
    nPtrj(1:MNvol + 1) = -1
    LHevalues = .false.
    LHevectors = .false.
    LHmatrix = .false.
    dpp = -1
    dqq = -1
    Lerrortype = 0
    Ngrid = -1
    dRZ = 1.0e-5_rkind
    Lcheck = 0
    Ltiming = .false.
    Ltransform = .false.
    fudge = 1.0e-00_rkind
    scaling = 1.0e-00_rkind
  end subroutine initialize_inputs
end module input_list_mod
