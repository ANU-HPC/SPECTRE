! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module input_parameters_mod
    use precisions_mod, only: rkind

    implicit none

    private
    public :: TInputParameters, TPhysicsParameters, TNumericParameters, TGlobalParameters, TDiagnosticsParameters, TScreenParameters

      ! The following parameters set the maximum allowed resolution:
    integer, parameter, public :: MNvol = 256 !< The maximum value of \c Nvol is \c MNvol=256.
    integer, parameter, public :: MMpol = 128 !< The maximum value of \c Mpol is \c MNpol=64.
    integer, parameter, public :: MNtor = 128 !< The maximum value of \c Ntor is \c MNtor=64.

    type :: TPhysicsParameters
        integer :: Igeometry = 3 !< selects Cartesian, cylindrical or toroidal geometry
        !< <ul>
        !< <li> \c Igeometry=1 : Cartesian: geometry determined by \f$R\f$: </li>
        !< <li> \c Igeometry=2 : cylindrical: geometry determined by \f$R\f$: </li>
        !< <li> \c Igeometry=3 : toroidal: geometry determined by \f$R\f$ *and* \f$Z\f$: </li>
        !< </ul>
        logical :: enforce_stell_sym = .true. !< stellarator symmetry is enforced if \c Istellsym is .true.
        integer :: Lfreebound = 0 !< compute vacuum field surrounding plasma
        real(rkind) :: phiedge = 1.0_rkind !< total enclosed toroidal magnetic flux
        real(rkind) :: curtor = 0.0_rkind !< total enclosed (toroidal) plasma current
        real(rkind) :: curpol = 0.0_rkind !< total enclosed (poloidal) linking current
        real(rkind) :: gamma = 0.0_rkind !< adiabatic index: cannot set \f$|\gamma| = 1\f$
        integer :: Nfp = 1 !< field periodicity
        !< <ul>
        !< <li> all Fourier representations are of the form \f$\cos(m\theta-n N \zeta)\f$, \f$\sin(m\theta-n N \zeta)\f$,
        !where \f$N\equiv\f$\c Nfp </li>
        !< <li> constraint: \c Nfp >= 1 </li>
        !< </ul>
        integer :: Nvol = 1 !< number of volumes
        !< <ul>
        !< <li> each volume \f${\cal V}_l\f$ is bounded by the \f${\cal I}_{l-1}\f$ and \f${\cal I}_{l}\f$ interfaces </li>
        !< <li> note that in cylindrical or toroidal geometry, \f${\cal I}_{0}\f$ is the degenerate coordinate axis </li>
        !< <li> constraint: \c Nvol<=MNvol </li>
        !< </ul>
        integer :: Mpol = 0 !< number of poloidal Fourier harmonics
        !< <ul>
        !< <li> all Fourier representations of doubly-periodic functions are of the form
        !< \f{eqnarray}{ f(\theta,\zeta) & = & \sum_{n=0}^{\texttt{Ntor}} f_{0,n}\cos(-n \, \texttt{Nfp} \, \zeta)
        !<                                   + \sum_{m=1}^{\texttt{Mpol}}\sum_{n=\texttt{-Ntor}}^{\texttt{Ntor}}
        !f_{m,n}\cos(m\theta-n \, \texttt{Nfp} \, \zeta),
        !< \f}
        !< Internally these "double" summations are written as a "single" summation,
        !< e.g. \f$f(\theta,\zeta) = \sum_j f_j \cos(m_j\theta-n_j\zeta)\f$. </li>
        !< </ul>
        integer :: Ntor = 0 !< number of toroidal Fourier harmonics
        !< <ul>
        !< <li> all Fourier representations of doubly-periodic functions are of the form
        !< \f{eqnarray}{ f(\theta,\zeta) & = & \sum_{n=0}^{\texttt{Ntor}} f_{0,n}\cos(-n \, \texttt{Nfp} \, \zeta)
        !<                                   + \sum_{m=1}^{\texttt{Mpol}}\sum_{n=\texttt{-Ntor}}^{\texttt{Ntor}}
        !f_{m,n}\cos(m\theta-n \, \texttt{Nfp} \, \zeta),
        !< \f}
        !< Internally these "double" summations are written as a "single" summation,
        !< e.g. \f$f(\theta,\zeta) = \sum_j f_j \cos(m_j\theta-n_j\zeta)\f$. </li>
        !< </ul>
        integer :: Lrad(1:MNvol + 1) = 4 !< Chebyshev resolution in each volume
        !< <ul>
        !< <li> constraint : \c Lrad(1:Mvol) >= 2 </li>
        !< </ul>
        integer :: Lconstraint = -1 !< selects constraints: primarily used in construct_beltrami_field() and solve_beltrami_system().
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
        !< <li> if \c Lconstraint==2, (UNTESTED!)  then in the plasma regions \f$\mu\f$ is adjusted
        !<      in order to satisfy the helicity constraint, \c helicity, at fixed enclosed fluxes;
        !<      and in the vacuum region \f$\Delta\psi_t\f$ and \f$\Delta\psi_p\f$ are varied to match the
        !<      prescribed plasma current, \c curtor, and the "linking" current, \c curpol, and \f$\mu = 0\f$
        !<      (the vacuum helicity is unconstrained). </li>
        !< <li> if \c Lconstraint.eq.3 , then the \f$\mu\f$ and \f$\psi_p\f$ variables are adjusted
        !<      in order to satisfy the volume and surface toroidal current computed with lbpol()
        !<      (excepted in the inner most volume, where the volume current is irrelevant).
        !<      Not implemented yet in free boundary.</li>
        !< </ul>
        real(rkind) :: tflux(1:MNvol + 1) = 0.0_rkind !< toroidal flux, \f$\psi_t\f$, enclosed by each interface
        !< <ul>
        !< <li> For each of the plasma volumes, this is a constraint: \c tflux is *not* varied </li>
        !< <li> For the vacuum region (only if \c Lfreebound==1), \c tflux  may be allowed to vary to match constraints </li>
        !< <li> Note that \c tflux  will be normalized so that \c tflux(Nvol) = 1.0,
        !<      so that \c tflux  is arbitrary up to a scale factor </li>
        !< <li> \sa phiedge </li>
        !< </ul>
        real(rkind) :: pflux(1:MNvol + 1) = 0.0_rkind !< poloidal flux, \f$\psi_p\f$, enclosed by each interface
        real(rkind) :: helicity(1:MNvol) = 0.0_rkind !< helicity, \f${\cal K}\f$, in each volume, \f${\cal V}_i\f$
        !< <ul>
        !< <li> on exit, \c helicity  is set to the computed values of \f${\cal K} \equiv \int {\bf A}\cdot{\bf B}\:dv\f$
        !</li>
        !< </ul>
        real(rkind) :: pscale = 0.0_rkind !< pressure scale factor
        !< <ul>
        !< <li> the initial pressure profile is given by \c pscale  \f$*\f$ \c pressure </li>
        !< </ul>
        real(rkind) :: pressure(1:MNvol + 1) = 0.0_rkind !< pressure in each volume
        !< <ul>
        !< <li> The pressure is *not* held constant, but \f$p_l V_l^\gamma = P_l\f$ *is* held constant,
        !<      where \f$P_l\f$ is determined by the initial pressures and the initial volumes, \f$V_l\f$. </li>
        !< <li> Note that if \c gamma==0.0, then \f$p_l \equiv P_l\f$. </li>
        !< <li> On output, the pressure is given by \f$p_l = P_l/V_l^\gamma\f$, where \f$V_l\f$ is the final volume. </li>
        !< <li> \c pressure is only used in calculation of interface force-balance. </li>
        !< </ul>
        integer :: Ladiabatic = 0 !< logical flag
        !< <ul>
        !< <li> If \c Ladiabatic==0, the adiabatic constants are determined by the initial pressure and volume. </li>
        !< <li> If \c Ladiabatic==1, the adiabatic constants are determined by the given input \c adiabatic. </li>
        !< </ul>
        real(rkind) :: adiabatic(1:MNvol + 1) = 0.0_rkind !< adiabatic constants in each volume
        !< <ul>
        !< <li> The pressure is *not* held constant, but \f$p_l V_l^\gamma = P_l \equiv\f$\c adiabatic is constant. </li>
        !< <li> Note that if \c gamma==0.0, then \c pressure==adiabatic. </li>
        !< <li> \c pressure is only used in calculation of interface force-balance. </li>
        !< </ul>
        real(rkind) :: mu(1:MNvol + 1) = 0.0_rkind !< helicity-multiplier, \f$\mu\f$, in each volume
        real(rkind) :: Ivolume(1:MNvol + 1) = 0.0_rkind !< Toroidal current constraint normalized by \f$\mu_0\f$
        !(\f$I_{volume} = \mu_0\cdot [A]\f$), in each volume.
        !< This is a cumulative quantity: \f$I_{\mathcal{V},i} = \int_0^{\psi_{t,i}} \mathbf{J}\cdot\mathbf{dS}\f$.
        !< Physically, it represents the sum of all non-pressure driven currents.
        real(rkind) :: Isurf(1:MNvol) = 0.0_rkind !< Toroidal current normalized by \f$\mu_0\f$ at each interface
        !(cumulative). This is the sum of all pressure driven currents.
        integer :: pl(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
        integer :: ql(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
        integer :: pr(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
        integer :: qr(0:MNvol) = 0 !< "inside" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !< where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2 \f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (inside) interface rotational-transform is defined by \c iota .
        real(rkind) :: iota(0:MNvol) = 0.0_rkind !< rotational-transform, \f$\mbox{$\,\iota\!\!$-}\f$, on inner side of each
        !interface
        !< <ul>
        !< <li> only relevant if illogical input for \c ql and \c qr are provided </li>
        !< </ul>
        integer :: lp(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
        integer :: lq(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
        integer :: rp(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
        integer :: rq(0:MNvol) = 0 !< "outer" interface rotational-transform is \f$\mbox{$\,\iota\!\!$-} = (p_l+\gamma
        !p_r)/(q_l+\gamma q_r)\f$,
        !<  where \f$\gamma\f$ is the golden mean, \f$\gamma = (1 + \sqrt 5 ) / 2\f$.
        !<
        !< If both \f$q_l = 0\f$ *and* \f$q_r = 0\f$, then the (outer) interface rotational-transform is defined by \c oita .
        real(rkind) :: oita(0:MNvol) = 0.0_rkind !< rotational-transform, \f$\mbox{$\,\iota\!\!$-}\f$, on outer side of each
        !interface
        !< <ul>
        !< <li> only relevant if illogical input for \c ql and \c qr are provided </li>
        !< </ul>
        real(rkind) :: mupftol = 1.0e-14_rkind !< accuracy to which \f$\mu\f$ and \f$\Delta\psi_p\f$ are required
        !< <ul>
        !< <li> only relevant if constraints on transform, enclosed currents etc. are to be satisfied iteratively, see \c
        !Lconstraint </li>
        !< </ul>
        integer :: mupfits = 8 !< an upper limit on the transform/helicity constraint iterations
        !< <ul>
        !< <li> only relevant if constraints on transform, enclosed currents etc. are to be satisfied iteratively, see \c
        !Lconstraint </li>
        !< <li> constraint: \c mupfits > 0 </li>
        !< </ul>
        real(rkind) :: rpol = 1.0_rkind !< poloidal extent of slab (effective radius)
        !< <ul>
        !< <li> only relevant if \c Igeometry==1 </li>
        !< <li> poloidal size is \f$L = 2\pi*\f$\c rpol </li>
        !< </ul>
        real(rkind) :: rtor = 1.0_rkind !< toroidal extent of slab (effective radius)
        !< <ul>
        !< <li> only relevant if \c Igeometry==1 </li>
        !< <li> toroidal size is \f$L = 2\pi*\f$\c rtor </li>
        !< </ul>
        integer :: Lreflect = 0 !< =1 reflect the upper and lower bound in slab, =0 do not reflect

        real(rkind) :: Rac(0:MNtor) = 0.0_rkind !<     stellarator symmetric coordinate axis
        real(rkind) :: Zas(0:MNtor) = 0.0_rkind !<     stellarator symmetric coordinate axis
        real(rkind) :: Ras(0:MNtor) = 0.0_rkind !< non-stellarator symmetric coordinate axis
        real(rkind) :: Zac(0:MNtor) = 0.0_rkind !< non-stellarator symmetric coordinate axis

        real(rkind) :: Rbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components
        real(rkind) :: Zbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components
        real(rkind) :: Rbs(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components
        real(rkind) :: Zbc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components

        real(rkind) :: Rwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components of wall
        real(rkind) :: Zws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric boundary components of wall
        real(rkind) :: Rws(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components of wall
        real(rkind) :: Zwc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric boundary components of wall

        real(rkind) :: Vns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric normal field at boundary:
        !vacuum component
        real(rkind) :: Bns(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !<     stellarator symmetric normal field at boundary:
        !plasma component
        real(rkind) :: Vnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric normal field at boundary:
        !vacuum component
        real(rkind) :: Bnc(-MNtor:MNtor, -MMpol:MMpol) = 0.0_rkind !< non-stellarator symmetric normal field at boundary:
        !plasma component

        contains

        procedure :: init => init_TPhysicsParameters
    end type TPhysicsParameters

    type :: TNumericParameters
        integer :: Linitialize = 0 !< Used to initialize geometry using a regularization / extrapolation method
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
        logical, public :: Lchangeangle = .false. !< flag to flip the direction of theta angle (to ensure right-handed R, phi, Z coordinate system)
        
        integer :: LautoinitBn = 1 !< Used to initialize \f$B_{ns}\f$ using an initial fixed-boundary calculation
        !< <ul>
        !< <li> only relevant if \c Lfreebound = 1 </li>
        !< <li> user-supplied \c Bns will only be considered if \c LautoinitBn = 0 </li>
        !< </ul>
        integer :: Lzerovac = 0 !< Used to adjust vacuum field to cancel plasma field on computational boundary
        !< <ul>
        !< <li> only relevant if \c Lfreebound = 1 </li>
        !< </ul>
        integer :: Ndiscrete = 2 !< resolution of the real space grid on which fast Fourier transforms are performed is given
        !by \c Ndiscrete*Mpol*4
        !< <ul>
        !< <li> constraint \c Ndiscrete>0 </li>
        !< </ul>
        integer :: Nquad = -1 !< Resolution of the Gaussian quadrature
        !< <ul>
        !< <li> The resolution of the Gaussian quadrature, \f$\displaystyle \int \!\! f(s) ds = \sum_k \omega_k f(s_k)\f$,
        !<      in each volume is given by \c Iquad\f$_v\f$,  </li>
        !< <li> \c Iquad\f$_v\f$ is set in initialize_internal_arrays() </li>
        !< </ul>
        integer :: iMpol = -4 !< Fourier resolution of straight-fieldline angle on interfaces
        !< <ul>
        !< <li> the rotational-transform on the interfaces is determined by a transformation to the straight-fieldline angle,
        !<      with poloidal resolution given by \c iMpol </li>
        !< <li> if \c iMpol<=0, then \c iMpol = Mpol - iMpol </li>
        !< </ul>
        integer :: iNtor = -4 !< Fourier resolution of straight-fieldline angle on interfaces
        !< <ul>
        !< <li> the rotational-transform on the interfaces is determined by a transformation to the straight-fieldline angle,
        !<      with toroidal resolution given by \c iNtor </li>
        !< <li> if \c iNtor<=0 then \c iNtor = Ntor - iNtor </li>
        !< <li> if \c Ntor==0, then the toroidal resolution of the angle transformation is set \c lNtor = 0 </li>
        !< </ul>
        integer :: Lsparse = 0 !< controls method used to solve for rotational-transform on interfaces
        !< <ul>
        !< <li> if \c Lsparse = 0, the transformation to the straight-fieldline angle is computed in Fourier space
        !<      using a dense matrix solver,  \c F04AAF </li>
        !< <li> if \c Lsparse = 1, the transformation to the straight-fieldline angle is computed in real space
        !<      using a dense matrix solver,  \c F04ATF </li>
        !< <li> if \c Lsparse = 2, the transformation to the straight-fieldline angle is computed in real space
        !<      using a sparse matrix solver, \c F11DEF </li>
        !< <li> if \c Lsparse = 3, the different methods for constructing the straight-fieldline angle are compared </li>
        !< </ul>
        integer :: Lsvdiota = 0 !< controls method used to solve for rotational-transform on interfaces
        !< only relevant if \c Lsparse = 0
        !< <ul>
        !< <li> if \c Lsvdiota = 0, use standard linear solver to construct straight fieldline angle transformation </li>
        !< <li> if \c Lsvdiota = 1, use SVD method to compute rotational-transform </li>
        !< </ul>
        integer :: Lrzaxis = 1 !< controls the guess of geometry axis in the innermost volume or initialization of interfaces
        !< <ul>
        !< <li> if \c Lrzaxis = 1, the centroid is used </li>
        !< <li> if \c Lrzaxis = 2, the Jacobian \f$m=1\f$ harmonic elimination method is used </li>
        !< </ul>
        integer :: Ntoraxis = 3 !< the number of \f$n\f$ harmonics used in the Jacobian \f$m=1\f$ harmonic elimination method
        !< only relevant if \c Lrzaxis.ge.1 .

        contains

        procedure :: init => init_TNumericParameters
    end type TNumericParameters

    type :: TGlobalParameters
        real(rkind) :: escale = 0.0_rkind !< controls the weight factor, \c BBweight, in the force-imbalance harmonics
        !< <ul>
        !< <li> \c BBweight(i) \f$\displaystyle \equiv \texttt{opsilon} \times \exp\left[-\texttt{escale} \times
        !(m_i^2+n_i^2) \right]\f$ </li>
        !< <li> defined in initialize_internal_arrays() : used in dforce() </li>
        !< <li> also see Eqn.\f$(\ref{eq:forcebalancemn_global})\f$ </li>
        !< </ul>
        real(rkind) :: opsilon = 1.0_rkind !< weighting of force-imbalance
        !< <ul>
        !< <li> used in dforce(): also see Eqn.\f$(\ref{eq:forcebalancemn_global})\f$ </li>
        !< </ul>
        real(rkind) :: pcondense = 2.0_rkind !< spectral condensation parameter
        !< <ul>
        !< <li> used in initialize_internal_arrays() to define \c mmpp(i) \f$\equiv m_i^p\f$, where \f$p\equiv \f$ \c pcondense </li>
        !< <li> the angle freedom is exploited to minimize \f$\displaystyle \texttt{epsilon} \sum_{i} m_i^p
        !(R_{i}^2+Z_{i}^2)\f$
        !<      with respect to tangential variations in the interface geometry </li>
        !< <li> also see Eqn.\f$(\ref{eq:spectralbalancemn_global})\f$ </li>
        !< </ul>
        real(rkind) :: epsilon = 0.0_rkind !< weighting of spectral-width constraint
        !< <ul>
        !< <li> used in dforce(): also see Eqn.\f$(\ref{eq:spectralbalancemn_global})\f$ </li>
        !< </ul>
        real(rkind) :: wpoloidal = 1.0_rkind !< "star-like" poloidal angle constraint radial exponential factor
        !< used in initialize_internal_arrays() to construct \c sweight
        real(rkind) :: upsilon = 1.0_rkind !< weighting of "star-like" poloidal angle constraint
        !< used in initialize_internal_arrays() to construct \c sweight
        integer :: mfreeits = 0 !< maximum allowed free-boundary iterations
        !< <ul>
        !< <li> only used if \c Lfreebound = 1 </li>
        !< <li> only used in xspech() </li>
        !< </ul>
        real(rkind) :: gBntol = 1.0e-06_rkind !< required tolerance in free-boundary iterations
        !< <ul>
        !< <li> only used if \c Lfreebound = 1 </li>
        !< <li> only used in xspech() </li>
        !< </ul>
        real(rkind) :: gBnbld = 0.666_rkind !< normal blend
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
        real(rkind) :: vcasingeps = 1.e-12_rkind !< regularization of Biot-Savart: see bnorml(), casing()
        real(rkind) :: vcasingtol = 1.e-08_rkind !< accuracy on virtual casing integral: see bnorml(), casing()
        integer :: vcasingits = 8 !< minimum number of calls to adaptive virtual casing routine: see casing()
        integer :: vcasingper = 1 !< periods of integragion  in adaptive virtual casing routine: see casing()

        contains

        procedure :: init => init_TGlobalParameters
    end type TGlobalParameters

    type :: TDiagnosticsParameters
        real(rkind) :: odetol = 1.0e-07_rkind !< o.d.e. integration tolerance for all field line tracing routines
        integer :: nPpts = 0 !< number of toroidal transits used (per trajectory) in following field lines
        !< for constructing Poincaré plots
        !< if \c nPpts<1, no Poincaré plot is constructed
        real(rkind) :: Ppts = 0.0_rkind !< stands for Poincare plot theta start. Chose at which angle (normalized over
        !\f$\pi\f$) the Poincare field-line tracing start.
        integer :: nPtrj(1:MNvol + 1) = -1 !< number of trajectories in each annulus to be followed in constructing Poincaré
        !plot
        !< <ul>
        !< <li> if \c nPtrj(l)<0, then \c nPtrj(l) = Ni(l),
        !<       where \c Ni(l) is the grid resolution used to construct the Beltrami field in volume \f$l\f$ </li>
        !< </ul>
        logical :: LHevalues = .false. !< to compute eigenvalues of \f$\nabla {\bf F}\f$
        logical :: LHevectors = .false. !< to compute eigenvectors (and also eigenvalues) of \f$\nabla {\bf F}\f$
        logical :: LHmatrix = .false. !< to compute and write to file the elements of \f$\nabla {\bf F}\f$
        integer :: Lerrortype = 0 !< the type of error output for Lcheck=1
        integer :: Ngrid = -1 !< the number of points to output in the grid, -1 for Lrad(vvol)
        integer :: Lcheck = 0 !< implement various checks
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
        !<      <li> set \c force_tol sufficiently small and set \c LreadGF = F,
        !<           so that the matrix of second derivatives is calculated </li>
        !<      <li> only for \c dspec executable, i.e. must compile with \c DFLAGS = "-D DEBUG" </li>
        !<      </ul> </li>
        !< <li> if \c Lcheck = 4, the analytic calculation of the derivatives of the magnetic field, \f$B^2\f$, at the
        !interfaces
        !<      is compared to a finite-difference estimate
        !<      <ul>
        !<      <li> set \c force_tol sufficiently small </li>
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
        !<      <li> must set \c Freebound=1 \c mfreeits!=0 </li>
        !<      <li> \c xdiagno must be executed manually </li>
        !<      </ul> </li>
        !< </ul>
        logical :: Ltransform = .false. !< to evaluate iota and straight field line coordinates

        contains

        procedure :: init => init_TDiagnosticsParameters
    end type TDiagnosticsParameters

    type :: TScreenParameters
        logical :: Wmanual = .false.
        logical :: Wrzaxis = .false.
        logical :: Wpackxi = .false.
        logical :: Wvolume = .false.
        logical :: Wcompute_coordinates = .false.
        logical :: Wbasefn = .false.
        logical :: Wmemory = .false.
        logical :: Wcompute_metric_quantities = .false.
        logical :: Wvolume_integrate_chebyshev = .false.
        logical :: Wmatrix = .false.
        logical :: Wspsint = .false.
        logical :: Wsolve_beltrami_system = .false.
        logical :: Wconstruct_beltrami_field = .false.
        logical :: Wpackab = .false.
        logical :: Wcompute_rotational_transform = .false.
        logical :: Wcompute_plasma_current = .false.
        logical :: Wdf00ab = .false.
        logical :: Wlforce = .false.
        logical :: Wintghs = .false.
        logical :: Wmtrxhs = .false.
        logical :: Wlbpol = .false.
        logical :: Wbroadcast_profile_data = .false.
        logical :: Wdfp100 = .false.
        logical :: Wdfp200 = .false.
        logical :: Wdforce = .false.
        logical :: Wnewton = .false.
        logical :: Wcasing = .false.
        logical :: Wbnorml = .false.
        logical :: Wcompute_beltrami_error = .false.
        logical :: Wconstruct_poincare_plot = .false.
        logical :: Wpp00ab = .false.
        logical :: Wget_position_vetor_x = .false.
        logical :: Whesian = .false.
        logical :: Wwrite_vector_potentials = .false.
        logical :: Wnumrec = .false.
        logical :: Wdcuhre = .false.
        logical :: Wminpack = .false.
        logical :: Wiqpack = .false.
        logical :: Wrksuite = .false.
        logical :: Wi1mach = .false.
        logical :: Wd1mach = .false.
        logical :: Wilut = .false.
        logical :: Witers = .false.
        logical :: Wsphdf5 = .false.
        logical :: Winitialize_internal_arrays = .false.
        logical :: Wglobal = .false.
        logical :: Wxspech = .false.
        logical :: Wbuild_vector_potential = .false. !< \todo: what is this?
        logical :: Wreadin = .false. !< write screen output of readin()
        logical :: Wwrtend = .false. !< write screen output of wrtend()
        logical :: Wmacros = .false. !< write screen output from expanded macros

        contains

        procedure :: init => init_TScreenParameters
    end type TScreenParameters

    type :: TInputParameters
        type(TPhysicsParameters) :: physics
        type(TNumericParameters) :: numeric
        type(TGlobalParameters) :: global
        type(TDiagnosticsParameters) :: diagnostics
        type(TScreenParameters) :: screen
    end type TInputParameters

    contains

    subroutine init_TPhysicsParameters(self, Igeometry, enforce_stell_sym, Lfreebound, phiedge, curtor, curpol, gamma, Nfp, Nvol, Mpol, Ntor, Lrad, Lconstraint, tflux, pflux, helicity, pscale, pressure, Ladiabatic, adiabatic, mu, Ivolume, Isurf, pl, ql, pr, qr, iota, lp, lq, rp, rq, oita, mupftol, mupfits, rpol, rtor, Lreflect, Rac, Zas, Ras, Zac, Rbc, Zbs, Rbs, Zbc, Rwc, Zws, Rws, Zwc, Vns, Bns, Vnc, Bnc)
        class(TPhysicsParameters), intent(inout) :: self
        integer, intent(in) :: Igeometry, Lfreebound, Nfp, Nvol, Mpol, Ntor, Lconstraint, Ladiabatic, mupfits, Lreflect
        logical, intent(in) :: enforce_stell_sym
        real(rkind), intent(in) :: phiedge, curtor, curpol, gamma, pscale, mupftol, rpol, rtor
        real(rkind), intent(in) :: tflux(1:MNvol + 1), pflux(1:MNvol + 1), helicity(1:MNvol), pressure(1:MNvol + 1), adiabatic(1:MNvol + 1), mu(1:MNvol + 1), Ivolume(1:MNvol + 1), Isurf(1:MNvol)
        integer, intent(in) :: Lrad(1:MNvol + 1), pl(0:MNvol), ql(0:MNvol), pr(0:MNvol), qr(0:MNvol), lp(0:MNvol), lq(0:MNvol), rp(0:MNvol), rq(0:MNvol)
        real(rkind), intent(in) :: iota(0:MNvol), oita(0:MNvol)
        real(rkind), intent(in) :: Rac(0:MNtor), Zas(0:MNtor), Ras(0:MNtor), Zac(0:MNtor)
        real(rkind), intent(in) :: Rbc(-MNtor:MNtor, -MMpol:MMpol), Zbs(-MNtor:MNtor, -MMpol:MMpol), Rbs(-MNtor:MNtor, -MMpol:MMpol), Zbc(-MNtor:MNtor, -MMpol:MMpol)
        real(rkind), intent(in) :: Rwc(-MNtor:MNtor, -MMpol:MMpol), Zws(-MNtor:MNtor, -MMpol:MMpol), Rws(-MNtor:MNtor, -MMpol:MMpol), Zwc(-MNtor:MNtor, -MMpol:MMpol)
        real(rkind), intent(in) :: Vns(-MNtor:MNtor, -MMpol:MMpol), Bns(-MNtor:MNtor, -MMpol:MMpol), Vnc(-MNtor:MNtor, -MMpol:MMpol), Bnc(-MNtor:MNtor, -MMpol:MMpol)

        self%Igeometry = Igeometry
        self%enforce_stell_sym = enforce_stell_sym
        self%Lfreebound = Lfreebound
        self%phiedge = phiedge
        self%curtor = curtor
        self%curpol = curpol
        self%gamma = gamma
        self%Nfp = Nfp
        self%Nvol = Nvol
        self%Mpol = Mpol
        self%Ntor = Ntor
        self%Lrad = Lrad
        self%Lconstraint = Lconstraint
        self%tflux = tflux
        self%pflux = pflux
        self%helicity = helicity
        self%pscale = pscale
        self%pressure = pressure
        self%Ladiabatic = Ladiabatic
        self%adiabatic = adiabatic
        self%mu = mu
        self%Ivolume = Ivolume
        self%Isurf = Isurf
        self%pl = pl
        self%ql = ql
        self%pr = pr
        self%qr = qr
        self%iota = iota
        self%lp = lp
        self%lq = lq
        self%rp = rp
        self%rq = rq
        self%oita = oita
        self%mupftol = mupftol
        self%mupfits = mupfits
        self%rpol = rpol
        self%rtor = rtor
        self%Lreflect = Lreflect
        self%Rac = Rac
        self%Zas = Zas
        self%Ras = Ras
        self%Zac = Zac
        self%Rbc = Rbc
        self%Zbs = Zbs
        self%Rbs = Rbs
        self%Zbc = Zbc
        self%Rwc = Rwc
        self%Zws = Zws
        self%Rws = Rws
        self%Zwc = Zwc
        self%Vns = Vns
        self%Bns = Bns
        self%Vnc = Vnc
        self%Bnc = Bnc
    end subroutine init_TPhysicsParameters

    subroutine init_TNumericParameters(self, Linitialize, Lchangeangle, LautoinitBn, Lzerovac, Ndiscrete, Nquad, iMpol, iNtor, Lsparse, Lsvdiota, Lrzaxis, Ntoraxis)
        class(TNumericParameters), intent(inout) :: self
        integer, intent(in) :: Linitialize, Lchangeangle, LautoinitBn, Lzerovac, Ndiscrete, Nquad, iMpol, iNtor, Lsparse, Lsvdiota, Lrzaxis, Ntoraxis

        self%Linitialize = Linitialize
        self%Lchangeangle = Lchangeangle
        self%LautoinitBn = LautoinitBn
        self%Lzerovac = Lzerovac
        self%Ndiscrete = Ndiscrete
        self%Nquad = Nquad
        self%iMpol = iMpol
        self%iNtor = iNtor
        self%Lsparse = Lsparse
        self%Lsvdiota = Lsvdiota
        self%Lrzaxis = Lrzaxis
        self%Ntoraxis = Ntoraxis
    end subroutine init_TNumericParameters

    subroutine init_TGlobalParameters(self, escale, opsilon, pcondense, epsilon, wpoloidal, upsilon, mfreeits, gBntol, gBnbld, vcasingeps, vcasingtol, vcasingits, vcasingper)
        class(TGlobalParameters), intent(inout) :: self
        integer, intent(in) :: mfreeits, vcasingits, vcasingper
        real(rkind), intent(in) :: escale, opsilon, pcondense, epsilon, wpoloidal, upsilon, gBntol, gBnbld, vcasingeps, vcasingtol

        self%escale = escale
        self%opsilon = opsilon
        self%pcondense = pcondense
        self%epsilon = epsilon
        self%wpoloidal = wpoloidal
        self%upsilon = upsilon
        self%mfreeits = mfreeits
        self%gBntol = gBntol
        self%gBnbld = gBnbld
        self%vcasingeps = vcasingeps
        self%vcasingtol = vcasingtol
        self%vcasingits = vcasingits
        self%vcasingper = vcasingper
    end subroutine init_TGlobalParameters

    subroutine init_TDiagnosticsParameters(self, odetol, nPpts, Ppts, nPtrj, LHevalues, LHevectors, LHmatrix, Lerrortype, Ngrid, Lcheck, Ltransform)
        class(TDiagnosticsParameters), intent(inout) :: self
        real(rkind), intent(in) :: odetol, Ppts
        integer, intent(in) :: nPpts, Lerrortype, Ngrid, Lcheck
        integer, intent(in) :: nPtrj(1:MNvol + 1)
        logical, intent(in) :: LHevalues, LHevectors, LHmatrix, Ltransform

        self%odetol = odetol
        self%nPpts = nPpts
        self%Ppts = Ppts
        self%nPtrj = nPtrj
        self%LHevalues = LHevalues
        self%LHevectors = LHevectors
        self%LHmatrix = LHmatrix
        self%Lerrortype = Lerrortype
        self%Ngrid = Ngrid
        self%Lcheck = Lcheck
        self%Ltransform = Ltransform
    end subroutine init_TDiagnosticsParameters

    subroutine init_TScreenParameters(self, Wmanual, Wrzaxis, Wpackxi, Wvolume, Wcompute_coordinates, Wbasefn, Wmemory, Wcompute_metric_quantities, Wvolume_integrate_chebyshev, Wmatrix, Wspsint, Wsolve_beltrami_system, Wconstruct_beltrami_field, Wpackab, Wcompute_rotational_transform, Wcompute_plasma_current, Wdf00ab, Wlforce, Wintghs, Wmtrxhs, Wlbpol, Wbroadcast_profile_data, Wdfp100, Wdfp200, Wdforce, Wnewton, Wcasing, Wbnorml, Wcompute_beltrami_error, Wconstruct_poincare_plot, Wpp00ab, Wget_position_vetor_x, Whesian, Wwrite_vector_potentials, Wnumrec, Wdcuhre, Wminpack, Wiqpack, Wrksuite, Wi1mach, Wd1mach, Wilut, Witers, Wsphdf5, Winitialize_internal_arrays, Wglobal, Wxspech, Wbuild_vector_potential, Wreadin, Wwrtend, Wmacros)
        class(TScreenParameters), intent(inout) :: self
        logical, intent(in) :: Wmanual, Wrzaxis, Wpackxi, Wvolume, Wcompute_coordinates, Wbasefn, Wmemory, Wcompute_metric_quantities, Wvolume_integrate_chebyshev, Wmatrix, Wspsint, Wsolve_beltrami_system, Wconstruct_beltrami_field, Wpackab, Wcompute_rotational_transform, Wcompute_plasma_current, Wdf00ab, Wlforce, Wintghs, Wmtrxhs, Wlbpol, Wbroadcast_profile_data, Wdfp100, Wdfp200, Wdforce, Wnewton, Wcasing, Wbnorml, Wcompute_beltrami_error, Wconstruct_poincare_plot, Wpp00ab, Wget_position_vetor_x, Whesian, Wwrite_vector_potentials, Wnumrec, Wdcuhre, Wminpack, Wiqpack, Wrksuite, Wi1mach, Wd1mach, Wilut, Witers, Wsphdf5, Winitialize_internal_arrays, Wglobal, Wxspech, Wbuild_vector_potential, Wreadin, Wwrtend, Wmacros

        self%Wmanual = Wmanual
        self%Wrzaxis = Wrzaxis
        self%Wpackxi = Wpackxi
        self%Wvolume = Wvolume
        self%Wcompute_coordinates = Wcompute_coordinates
        self%Wbasefn = Wbasefn
        self%Wmemory = Wmemory
        self%Wcompute_metric_quantities = Wcompute_metric_quantities
        self%Wvolume_integrate_chebyshev = Wvolume_integrate_chebyshev
        self%Wmatrix = Wmatrix
        self%Wspsint = Wspsint
        self%Wsolve_beltrami_system = Wsolve_beltrami_system
        self%Wconstruct_beltrami_field = Wconstruct_beltrami_field
        self%Wpackab = Wpackab
        self%Wcompute_rotational_transform = Wcompute_rotational_transform
        self%Wcompute_plasma_current = Wcompute_plasma_current
        self%Wdf00ab = Wdf00ab
        self%Wlforce = Wlforce
        self%Wintghs = Wintghs
        self%Wmtrxhs = Wmtrxhs
        self%Wlbpol = Wlbpol
        self%Wbroadcast_profile_data = Wbroadcast_profile_data
        self%Wdfp100 = Wdfp100
        self%Wdfp200 = Wdfp200
        self%Wdforce = Wdforce
        self%Wnewton = Wnewton
        self%Wcasing = Wcasing
        self%Wbnorml = Wbnorml
        self%Wcompute_beltrami_error = Wcompute_beltrami_error
        self%Wconstruct_poincare_plot = Wconstruct_poincare_plot
        self%Wpp00ab = Wpp00ab
        self%Wget_position_vetor_x = Wget_position_vetor_x
        self%Whesian = Whesian
        self%Wwrite_vector_potentials = Wwrite_vector_potentials
        self%Wnumrec = Wnumrec
        self%Wdcuhre = Wdcuhre
        self%Wminpack = Wminpack
        self%Wiqpack = Wiqpack
        self%Wrksuite = Wrksuite
        self%Wi1mach = Wi1mach
        self%Wd1mach = Wd1mach
        self%Wilut = Wilut
        self%Witers = Witers
        self%Wsphdf5 = Wsphdf5
        self%Winitialize_internal_arrays = Winitialize_internal_arrays
        self%Wglobal = Wglobal
        self%Wxspech = Wxspech
        self%Wbuild_vector_potential = Wbuild_vector_potential
        self%Wreadin = Wreadin
        self%Wwrtend = Wwrtend
        self%Wmacros = Wmacros
    end subroutine init_TScreenParameters
end module input_parameters_mod
