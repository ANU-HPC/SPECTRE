! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module cputiming_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private

  real(rkind), public :: Tmanual = 0.0_rkind, manualT = 0.0_rkind
  real(rkind), public :: Trzaxis = 0.0_rkind, rzaxisT = 0.0_rkind
  real(rkind), public :: Tpackxi = 0.0_rkind, packxiT = 0.0_rkind
  real(rkind), public :: Tvolume = 0.0_rkind, volumeT = 0.0_rkind
  real(rkind), public :: Tcompute_coordinates = 0.0_rkind, compute_coordinatesT = 0.0_rkind
  real(rkind), public :: Tbasefn = 0.0_rkind, basefnT = 0.0_rkind
  real(rkind), public :: Tmemory = 0.0_rkind, memoryT = 0.0_rkind
  real(rkind), public :: Tcompute_metric_quantities = 0.0_rkind, compute_metric_quantitiesT = 0.0_rkind
  real(rkind), public :: Tvolume_integrate_chebyshev = 0.0_rkind, volume_integrate_chebyshevT = 0.0_rkind
  real(rkind), public :: Tmatrix = 0.0_rkind, matrixT = 0.0_rkind
  real(rkind), public :: Tspsint = 0.0_rkind, spsintT = 0.0_rkind
  real(rkind), public :: Tsolve_beltrami_system = 0.0_rkind, solve_beltrami_systemT = 0.0_rkind
  real(rkind), public :: Tconstruct_beltrami_field = 0.0_rkind, construct_beltrami_fieldT = 0.0_rkind
  real(rkind), public :: Tpackab = 0.0_rkind, packabT = 0.0_rkind
  real(rkind), public :: Tcompute_rotational_transform = 0.0_rkind, compute_rotational_transformT = 0.0_rkind
  real(rkind), public :: Tcompute_plasma_current = 0.0_rkind, compute_plasma_currentT = 0.0_rkind
  real(rkind), public :: Tdf00ab = 0.0_rkind, df00abT = 0.0_rkind
  real(rkind), public :: Tlforce = 0.0_rkind, lforceT = 0.0_rkind
  real(rkind), public :: Tintghs = 0.0_rkind, intghsT = 0.0_rkind
  real(rkind), public :: Tmtrxhs = 0.0_rkind, mtrxhsT = 0.0_rkind
  real(rkind), public :: Tlbpol = 0.0_rkind, lbpolT = 0.0_rkind
  real(rkind), public :: Tbroadcast_profile_data = 0.0_rkind, broadcast_profile_dataT = 0.0_rkind
  real(rkind), public :: Tdfp100 = 0.0_rkind, dfp100T = 0.0_rkind
  real(rkind), public :: Tdfp200 = 0.0_rkind, dfp200T = 0.0_rkind
  real(rkind), public :: Tdforce = 0.0_rkind, dforceT = 0.0_rkind
  real(rkind), public :: Tnewton = 0.0_rkind, newtonT = 0.0_rkind
  real(rkind), public :: Tcasing = 0.0_rkind, casingT = 0.0_rkind
  real(rkind), public :: Tbnorml = 0.0_rkind, bnormlT = 0.0_rkind
  real(rkind), public :: Tcompute_beltrami_error = 0.0_rkind, compute_beltrami_errorT = 0.0_rkind
  real(rkind), public :: Tconstruct_poincare_plot = 0.0_rkind, construct_poincare_plotT = 0.0_rkind
  real(rkind), public :: Tpp00ab = 0.0_rkind, pp00abT = 0.0_rkind
  real(rkind), public :: Tcompute_magnetic_field = 0.0_rkind, compute_magnetic_fieldT = 0.0_rkind
  real(rkind), public :: Tget_position_vetor_x = 0.0_rkind, get_position_vetor_xT = 0.0_rkind
  real(rkind), public :: Thesian = 0.0_rkind, hesianT = 0.0_rkind
  real(rkind), public :: Twrite_vector_potentials = 0.0_rkind, write_vector_potentialsT = 0.0_rkind
  real(rkind), public :: Tnumrec = 0.0_rkind, numrecT = 0.0_rkind
  real(rkind), public :: Tdcuhre = 0.0_rkind, dcuhreT = 0.0_rkind
  real(rkind), public :: Tminpack = 0.0_rkind, minpackT = 0.0_rkind
  real(rkind), public :: Tiqpack = 0.0_rkind, iqpackT = 0.0_rkind
  real(rkind), public :: Trksuite = 0.0_rkind, rksuiteT = 0.0_rkind
  real(rkind), public :: Ti1mach = 0.0_rkind, i1machT = 0.0_rkind
  real(rkind), public :: Td1mach = 0.0_rkind, d1machT = 0.0_rkind
  real(rkind), public :: Tilut = 0.0_rkind, ilutT = 0.0_rkind
  real(rkind), public :: Titers = 0.0_rkind, itersT = 0.0_rkind
  real(rkind), public :: Tsphdf5 = 0.0_rkind, sphdf5T = 0.0_rkind
  real(rkind), public :: Tinitialize_internal_arrays = 0.0_rkind, initialize_internal_arraysT = 0.0_rkind
  real(rkind), public :: Tglobal = 0.0_rkind, globalT = 0.0_rkind
  real(rkind), public :: Tinputlist = 0.0_rkind, inputlistT = 0.0_rkind
end module cputiming_mod
