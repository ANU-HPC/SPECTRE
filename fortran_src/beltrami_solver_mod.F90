! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module beltrami_solver_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind, machprec
  use io_mod, only: ounit
  use input_list_mod, only: Wmacros, Wsolve_beltrami_system, Wcompute_rotational_transform, Wcompute_plasma_current, &
      Wconstruct_beltrami_field, mu, helicity, &
      iota, oita, curtor, curpol, Lrad, Ntor, Lconstraint, mupftol, enforce_stell_sym
  use allglobal_mod, only: myid, cpus, ivol, Lcoordinatesingularity, Lplasmaregion, mn, mns, Nt, Nz, NAdof, dMA, dMB, &
      dMD, dMG, solution, dtflux, dpflux, diotadxup, dItGpdxtp, lBBintegral, lABintegral, xoffset, ImagneticOK, mvol
  use memory_allocation_mod, only: sallocate
  use errors_mod, only: fatal
  use plasma_current_mod, only: compute_plasma_current
  use packing_mod, only: packab
  use magnetic_field_mod, only: compute_rotational_transform

  implicit none(type, external)

  private
  public :: solve_beltrami_system

  external :: dcopy, DGETRF, DGECON, dgerfs, DGETRS

contains

  subroutine solve_beltrami_system(Ndof, Xdof, Fdof, Ddof, Ldfjac, iflag)
    integer, intent(in) :: Ndof
    integer, intent(in) :: Ldfjac
    real(rkind), intent(in) :: Xdof(1:Ndof)
    real(rkind), intent(inout) :: Fdof(1:Ndof)
    real(rkind), intent(inout) :: Ddof(1:Ldfjac, 1:Ndof)
    integer, intent(inout) :: iflag

    real(rkind) :: cput
    integer, parameter :: NB = 4 ! optimal workspace block size for LAPACK:DGECON
    integer :: lvol, NN, MM, ideriv, lmns, Lwork
    integer :: idgetrf(0:1), idgetrs(0:1), idgerfs(0:1), idgecon(0:1)
    real(rkind) :: lmu, dpf, dtf, dpsi(1:2), tpsi(1:2), ppsi(1:2), lcpu
    real(rkind) :: anorm, rcond, ferr(2), berr(2)
    character :: packorunpack

    ! For direct LU decompose
    integer, allocatable :: ipiv(:), Iwork(:)
    real(rkind), allocatable :: matrix(:, :), rhs(:, :), LU(:, :)
    real(rkind), allocatable :: RW(:), RD(:, :)

    lvol = ivol ! recall that ivol is global

    if (Lplasmaregion) then
      lmu = Xdof(1) - xoffset
      dtf = dtflux(lvol)
      if (Ndof == 2) then
        dpf = Xdof(2) - xoffset
      else
        dpf = dpflux(lvol)
      end if

      ! only toroidal flux is modified
      if (Lconstraint == -2) then
        lmu = mu(lvol)
        dpf = dpflux(lvol)
        if (lvol == mvol) then
          dtf = Xdof(1) - xoffset
        else
          dtf = dtflux(lvol)
        end if
      end if

    else ! Lvacuumregion
      lmu = 0.0_rkind ! restrict attention to strict vacuum field
      dtf = Xdof(1) - xoffset
      if (Ndof == 2) then
        dpf = Xdof(2) - xoffset
      else
        dpf = dpflux(lvol)
      end if

    end if

    dpsi(1:2) = [dtf, dpf] ! enclosed poloidal fluxes and their derivatives
    tpsi(1:2) = [1.0_rkind, 0.0_rkind] ! enclosed toroidal fluxes and their derivatives
    ppsi(1:2) = [0.0_rkind, 1.0_rkind] ! enclosed toroidal fluxes and their derivatives

    diotadxup(0:1, -1:2, lvol) = 0.0_rkind ! rotational-transform, and its derivatives with respect to lmu and dpf, or
    ! toroidal current, on the inner/outer interface
    dItGpdxtp(0:1, -1:2, lvol) = 0.0_rkind ! plasma and linking currents

    NN = NAdof(lvol)! shorthand

    call sallocate(rhs, 1, NN, 0, 2, 0.0_rkind)
    call sallocate(matrix, 1, NN, 1, NN, 0.0_rkind)

    solution(1:NN, -1:2) = 0.0_rkind ! this is a global array allocated in dforce

    ! allocate work space
    Lwork = NB * NN

    call sallocate(RW, 1, Lwork, 0.0_rkind)
    call sallocate(RD, 1, NN, 0, 2, 0.0_rkind)
    call sallocate(LU, 1, NN, 1, NN, 0.0_rkind)
    call sallocate(ipiv, 1, NN, 0)
    call sallocate(Iwork, 1, NN, 0)

    idgetrf(0:1) = 0 ! error flags
    idgetrs(0:1) = 0 ! error flags
    idgerfs(0:1) = 0 ! error flags
    idgecon(0:1) = 0 ! error flags

    ! ideriv labels derivative as follows:
    ! ideriv = 0 : compute Beltrami field: ideriv = 1 : compute d Beltrami field / d \mu           : ideriv = 2 : compute
    ! d Beltrami field / d \Delta \psi_p
    ! ideriv = 0 : compute Vacuum   field: ideriv = 1 : compute d Vacuum   field / d \Delta \psi_t : ideriv = 2 : compute
    ! d Vacuum   field / d \Delta \psi_p

    do ideriv = 0, 1 ! loop over derivatives
      if (iflag == 1 .and. ideriv == 1) cycle ! only need to return function: recall the derivative estimate requires
      ! function evaluation

      if (Lcoordinatesingularity) then
        matrix(1:NN, 1:NN) = dMA(1:NN, 1:NN) - lmu * dMD(1:NN, 1:NN)

        select case (ideriv)
        case (0)
          rhs(1:NN, 0) = -matmul(dMB(1:NN, 1:2), dpsi(1:2))
        case (1) ! construct dMD*solution

          rhs(1:NN, 1) = -matmul(-1.0_rkind * dMD(1:NN, 1:NN), solution(1:NN, 0))
          rhs(1:NN, 2) = -matmul(dMB(1:NN, 1:2), ppsi(1:2))
        end select

        if (Lconstraint == -2) then
          select case (ideriv)
          case (0)
            rhs(1:NN, 0) = -dMG(1:NN) - matmul(dMB(1:NN, 1:2), dpsi(1:2))
          case (1)
            rhs(1:NN, 1) = -matmul(dMB(1:NN, 1:2), tpsi(1:2))
          end select
        end if

      else ! .not.Lcoordinatesingularity

        if (Lplasmaregion) then
          matrix(1:NN, 1:NN) = dMA(1:NN, 1:NN) - lmu * dMD(1:NN, 1:NN)

          select case (ideriv)
          case (0)
            rhs(1:NN, 0) = -matmul(dMB(1:NN, 1:2), dpsi(1:2))
          case (1) ! construct dMD*solution
            rhs(1:NN, 1) = -matmul(-1.0_rkind * dMD(1:NN, 1:NN), solution(1:NN, 0))
            rhs(1:NN, 2) = -matmul(dMB(1:NN, 1:2), ppsi(1:2))
          end select

        else ! Lvacuumregion
          matrix(1:NN, 1:NN) = dMA(1:NN, 1:NN)! - lmu * dMD(1:NN,1:NN)

          select case (ideriv)
          case (0)
            rhs(1:NN, 0) = -dMG(1:NN) - matmul(dMB(1:NN, 1:2), dpsi(1:2))! perhaps there is an lmu term missing here
          case (1)
            rhs(1:NN, 1) = -matmul(dMB(1:NN, 1:2), tpsi(1:2))! perhaps there is an lmu term missing here
            rhs(1:NN, 2) = -matmul(dMB(1:NN, 1:2), ppsi(1:2))! perhaps there is an lmu term missing here
          end select
        end if
      end if

      select case (ideriv)
      case (0) ! ideriv=0
        MM = 1
        !LU = matrix
        call DCOPY(NN * NN, matrix, 1, LU, 1)! BLAS version
        solution(1:NN, 0) = rhs(:, 0)
        call DGETRF(NN, NN, LU, NN, ipiv, idgetrf(ideriv))! LU factorization

        anorm = maxval(sum(abs(matrix), 1))
        call DGECON('I', NN, LU, NN, anorm, rcond, RW, Iwork, idgecon(ideriv))! estimate the condition number

        call DGETRS('N', NN, MM, LU, NN, ipiv, solution(1:NN, 0), NN, idgetrs(ideriv))! sovle linear equation
        call DGERFS('N', NN, MM, matrix, NN, LU, NN, ipiv, rhs(1, 0), NN, solution(1, 0), NN, FERR, BERR, RW, Iwork,&
            & idgerfs(ideriv))! refine the solution

      case (1) ! ideriv=1
        MM = 2
        solution(1:NN, 1:MM) = rhs(:, 1:MM)
        call DGETRS('N', NN, MM, LU, NN, ipiv, solution(1:NN, 1:MM), NN, idgetrs(ideriv))
        call DGERFS('N', NN, MM, matrix, NN, LU, NN, ipiv, rhs(1, 1), NN, solution(1, 1), NN, FERR, BERR, RW, Iwork,&
            & idgerfs(ideriv))
      end select ! ideriv

      cput = MPI_Wtime()

      if (idgetrf(ideriv) == 0 .and. idgetrs(ideriv) == 0 .and. idgerfs(ideriv) == 0 .and. rcond >= machprec) then
        if (Wsolve_beltrami_system) write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), &
            idgetrs(ideriv), idgetrf(ideriv), "success", cput - lcpu
      elseif (idgetrf(ideriv) < 0 .or. idgetrs(ideriv) < 0 .or. idgerfs(ideriv) < 0) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "input error"
      elseif (idgetrf(ideriv) > 0) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "singular"
      elseif (rcond <= machprec) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "ill conditioned"
      else
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "invalid error"
      end if

      1010 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; ideriv=", i2, " ; ", a23, "=", i3, ' ', &
          i3, ' ', i3, " ;", a34, :, " time=", f10.2, " ")
    end do

    do ideriv = 0, 2
      if (iflag == 1 .and. ideriv > 0) cycle

      packorunpack = 'U'
      call packab(packorunpack, lvol, NN, solution(1:NN, ideriv), ideriv)! unpacking: this assigns oAt, oAz through
      ! common
    end do ! do ideriv = 0, 2

    ! can compute the energy and helicity integrals: easiest to do this with solution in packed format
    lBBintegral(lvol) = 1.0_rkind / 2.0_rkind * sum(solution(1:NN, 0) * matmul(dMA(1:NN, 1:NN), solution(1:NN, 0))) +&
        & sum(solution(1:NN, 0) * matmul(dMB(1:NN, 1:2), dpsi(1:2)))

    lABintegral(lvol) = 1.0_rkind / 2.0_rkind * sum(solution(1:NN, 0) * matmul(dMD(1:NN, 1:NN), solution(1:NN, 0)))

    deallocate(matrix)
    deallocate(rhs)
    deallocate(RW)
    deallocate(RD)
    deallocate(LU)
    deallocate(ipiv)
    deallocate(Iwork)

    idgetrf(0:1) = abs(idgetrf(0:1)) + abs(idgetrs(0:1)) + abs(idgerfs(0:1)) + abs(idgecon(0:1))
    if (idgetrf(0)/= 0 .or. idgetrf(1)/= 0) then ! failed to construct Beltrami/vacuum field and/or derivatives

      ImagneticOK(lvol) = .false. ! set error flag

      if (iflag == 1) Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out
      if (iflag == 2) Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      iflag = -1 ! this value will be returned by C05PCF to construct_beltrami_field

      return

    else

      ImagneticOK(lvol) = .true. ! set error flag: used in dforce

    end if

    if (enforce_stell_sym) then
      lmns = 1 + (mns - 1) ! number of independent degrees of freedom in angle transformation
    else
      lmns = 1 + (mns - 1) + (mns - 1) ! only required for dense, Fourier angle transformation
    end if

    select case (Lconstraint)

    case (-2 ) ! Lconstraint=-2
      call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

      ! Constraint on curpol only
      if ( iflag == 1 ) Fdof(1) = dItGpdxtp(1, 0, lvol) - curpol

      ! Derivative w.r.t. toroidal flux only
      if ( iflag == 2 ) Ddof(1, 1) = dItGpdxtp(1, 1, lvol)
    case (-1) ! Lconstraint=-1

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: Lconstraint=-1 indicates no iterations over mu   , dpflux
        ! are required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        if (Wcompute_plasma_current) then ! compute enclosed currents    only for diagnostic purposes
          call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)
          curtor = dItGpdxtp(0, 0, lvol)! icurrent(0) ! update input variables
          curpol = dItGpdxtp(1, 0, lvol)! gcurrent(0)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out:Lconstraint=-1 indicates no iterations over dtflux, dpflux
        ! are required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      end if

    case (0) ! Lconstraint= 0

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: Lconstraint= 0 indicates no iterations over mu, dpflux are
        ! required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

        if (iflag == 1) Fdof(1:2) = [dItGpdxtp(0, 0, lvol) - curtor, dItGpdxtp(1, 0, lvol) - curpol]
        if (iflag == 2) Ddof(1:2, 1) = [dItGpdxtp(0, 1, lvol), dItGpdxtp(1, 1, lvol)]
        !if( iflag.eq.2 ) Ddof(1:2,2) = (/ dItGpdxtp(0,0,lvol)         , dItGpdxtp(1,2,lvol)          /)
        if (iflag == 2) Ddof(1:2, 2) = [dItGpdxtp(0, 2, lvol), dItGpdxtp(1, 2, lvol)]

      end if

    case (1) ! Lconstraint= 1

      call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)! required for both plasma and vacuum region

      if (Lplasmaregion) then

        if (Lcoordinatesingularity) then ! Ndof = 1
          if (iflag == 1) Fdof(1) = diotadxup(1, 0, lvol) - iota(lvol)
          if (iflag == 2) Ddof(1, 1) = diotadxup(1, 1, lvol)! derivative of outer rotational-transform wrt helicity
          ! multiplier
        end if

        if (Ndof == 2) then
          if (iflag == 1) Fdof(1:2) = diotadxup(0:1, 0, lvol) - [oita(lvol - 1), iota(lvol)]
          if (iflag == 2) Ddof(1:2, 1) = diotadxup(0:1, 1, lvol)
          if (iflag == 2) Ddof(1:2, 2) = diotadxup(0:1, 2, lvol)
        end if

      else ! Lvacuumregion

        call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

        curtor = dItGpdxtp(0, 0, lvol)! update input variables
        !curpol = dItGpdxtp(1,0,lvol)

        if (iflag == 1) Fdof(1:2) = [diotadxup(0, 0, lvol) - oita(lvol - 1), dItGpdxtp(1, 0, lvol) - curpol]
        if (iflag == 2) Ddof(1:2, 1) = [diotadxup(0, 1, lvol), dItGpdxtp(1, 1, lvol)]
        if (iflag == 2) Ddof(1:2, 2) = [diotadxup(0, 2, lvol), dItGpdxtp(1, 2, lvol)]

      end if

    case (2)

      if (iflag == 1) Fdof(1) = lABintegral(lvol) - helicity(lvol)

      if (iflag == 2) Ddof(1, 1) = 1.0_rkind / 2.0_rkind * sum(solution(1:NN, 1) * matmul(dMD(1:NN, 1:NN), solution(1:NN,&
          & 0))) + 1.0_rkind / 2.0_rkind * sum(solution(1:NN, 0) * matmul(dMD(1:NN, 1:NN), solution(1:NN, 1)))

    case (3)

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: no iteration other mu and psip locally
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        !WCALL( solve_beltrami_system, compute_plasma_current,( lvol, mn, Nt, Nz, iflag, dItGpdxtp(0:1,-1:2,lvol) ) )

        ! Iteration only on toroidal flux.
        ! if( iflag.eq.1 ) Fdof(1:Ndof  ) = dItGpdxtp(1,0,lvol) - curpol
        ! if( iflag.eq.2 ) Ddof(1:Ndof,1:Ndof) = dItGpdxtp(1,1,lvol)

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: no iteration other mu and psip locally
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      end if

    end select

    if (Wsolve_beltrami_system .or. Wconstruct_beltrami_field) then ! the following is screen output

      cput = MPI_Wtime()

      if (Lplasmaregion) then
        select case (iflag)
        case (0)
          write(ounit, 3000)cput - cpus, myid, lvol, lmu, dpf, iflag ! this is impossible by above logic
        case (1)
          write(ounit, 3000)cput - cpus, myid, lvol, lmu, dpf, iflag, Fdof(1:Ndof)
        case (2)
          write(ounit, 3010)cput - cpus, myid, lvol, lmu, dpf, iflag, Ddof(1:Ndof, 1:Ndof)
        case default
          call fatal(.true., __FILE__, "illegal iflag on entry ")

        end select
      else ! Lvacuumregion
        select case (iflag)
        case (0)
          write(ounit, 3001)cput - cpus, myid, lvol, dtf, dpf, iflag ! this is impossible by above logic
        case (1)
          write(ounit, 3001)cput - cpus, myid, lvol, dtf, dpf, iflag, Fdof(1:Ndof)
        case (2)
          write(ounit, 3011)cput - cpus, myid, lvol, dtf, dpf, iflag, Ddof(1:Ndof, 1:Ndof)
        case default
          call fatal(.true., __FILE__, "illegal iflag on entry ")

        end select
      end if

      3000 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (mu,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " F=", 2es13.05, " ")
      3010 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (mu,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " D=", 4es13.05, " ")
      3001 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (dt,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " F=", 2es13.05, " ")
      3011 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (dt,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " D=", 4es13.05, " ")

    end if

    if (iflag == 1) then ! only in this case is Fdof defined

      if (sum(abs(Fdof(1:Ndof))) / Ndof < mupftol) then ! satisfactory

        if (Lplasmaregion) then
          mu(lvol) = lmu

          dpflux(lvol) = dpf
        else
          mu(lvol) = 0.0_rkind
          dtflux(lvol) = dtf
          dpflux(lvol) = dpf
        end if

        if ( Lconstraint == -2) then
          !mu(lvol) = lmu
          dtflux(lvol) = dtf
        end if

        iflag = -2 ! return "acceptance" flag through to construct_beltrami_field via ifail: early termination
      end if
    end if
  end subroutine solve_beltrami_system
end module beltrami_solver_mod
