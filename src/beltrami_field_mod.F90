module beltrami_field_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind
  use io_mod, only: stdout
  use input_list_mod, only: Wmacros, Wconstruct_beltrami_field, Lconstraint, mu, helicity, mupftol, mupfits, Lrad, Lcheck
  use cputiming_mod
  use allglobal_mod, only: myid, cpus, lABintegral, ivol, dtflux, dpflux, xoffset, Lcoordinatesingularity, Lplasmaregion, &
      Lvacuumregion
  use errors_mod, only: fatal
  use beltrami_solver_mod, only:solve_beltrami_system

  implicit none(type, external)

  private
  public :: construct_beltrami_field

  external :: hybrj2

contains

  subroutine construct_beltrami_field(lvol, NN)
    integer, intent(in) :: lvol
    integer, intent(in) :: NN

    real(rkind) :: cput
    real(rkind) :: tol, lastcpu
    integer :: Nxdof, Ndof, Ldfjac, iflag, maxfev, mode, LRR, nfev, njev, nprint, ihybrj
    real(rkind) :: Xdof(1:2), Fdof(1:2), Ddof(1:2, 1:2)
    real(rkind) :: factor, diag(1:2), RR(1:2 * (2 + 1) / 2), QTF(1:2), wk(1:2, 1:4)

    ivol = lvol ! various subroutines (e.g. solve_beltrami_system, df00ab) that may be called below require volume identification, but
    ! the argument list is fixed by NAG

    lastcpu = MPI_Wtime()

    if (Lplasmaregion) then
      Xdof(1:2) = xoffset + [mu(lvol), dpflux(lvol)] ! initial guess for degrees of freedom: offset from zero so that
      ! relative error is small

      if (Lconstraint == -2) then
        Xdof(1:1) = xoffset + [ dtflux(lvol)]
      end if

      select case (Lconstraint)
      case (-2 )
        Nxdof = 1 ! toroidal flux IS varied to match linking current;
      case (-1)
        Nxdof = 0 ! multiplier & poloidal flux NOT varied
        iflag = 1 !(we don't need derivatives)
      case (0)
        Nxdof = 0 ! multiplier & poloidal flux NOT varied
        iflag = 1 !(we don't need derivatives)
      case (1)
        if (Lcoordinatesingularity) then
          Nxdof = 1 ! multiplier                 IS  varied to match       outer transform
        else
          Nxdof = 2 ! multiplier & poloidal flux ARE varied to match inner/outer transform
        end if
      case (2)
        Nxdof = 1 ! multiplier                 IS  varied to match             helicity
      case (3)
        if (Lcoordinatesingularity) then
          Nxdof = 0 ! multiplier & poloidal flux NOT varied
        else
          Nxdof = 0 ! Global constraint, no dof locally
        end if
        iflag = 2 !(we still need derivatives)
      end select
    else ! Lvacuumregion
      Xdof(1:2) = xoffset + [dtflux(lvol), dpflux(lvol)] ! initial guess for degrees of freedom: offset from zero so
      ! that relative error is small

      select case (Lconstraint)
      case (-1)
        Nxdof = 0 ! poloidal   & toroidal flux NOT varied

      case (0)
        Nxdof = 2 ! poloidal   & toroidal flux ARE varied to match linking current and plasma current

      case (1)
        Nxdof = 2 ! poloidal   & toroidal flux ARE varied to match linking current and transform-constraint

      case (2)
        Nxdof = 2 ! poloidal   & toroidal flux ARE varied to match linking current and plasma current

      case (3)
        Nxdof = 0 ! Fluxes are determined in dforce via a linear system
        iflag = 2
      end select
    end if
    select case (Nxdof)

    case (0) ! need only call solve_beltrami_system once, to calculate Beltrami field for given helicity multiplier and enclosed fluxes
      Ndof = 1
      Ldfjac = Ndof
      nfev = 1
      njev = 0
      ihybrj = 1
      ! provide dummy values for consistency

      call solve_beltrami_system(Ndof, Xdof(1:Ndof), Fdof(1:Ndof), Ddof(1:Ldfjac, 1:Ndof), Ldfjac, iflag)

      helicity(lvol) = lABintegral(lvol)! this was computed in solve_beltrami_system

    case (1:2) ! will iteratively call solve_beltrami_system, to calculate Beltrami field that satisfies constraints
      Ndof = Nxdof
      Ldfjac = Ndof
      nfev = 0
      njev = 0
      ihybrj = 0
      tol = mupftol
      LRR = Ndof * (Ndof + 1) / 2
      mode = 0
      diag(1:2) = 0.0_rkind
      factor = 1.0_rkind
      maxfev = mupfits
      nprint = 0

      call fatal(Ndof > 2, __FILE__, "illegal ")

      call hybrj2(solve_beltrami_system, Ndof, Xdof(1:Ndof), Fdof(1:Ndof), Ddof(1:Ldfjac, 1:Ndof), Ldfjac, tol, maxfev, diag(1:Ndof&
          ),&
          & mode, factor, nprint, ihybrj, nfev, njev, RR(1:LRR), LRR, QTF(1:Ndof), WK(1:Ndof, 1), WK(1:Ndof, 2),&
          & WK(1:Ndof, 3), WK(1:Ndof, 4))

      if (Lplasmaregion) then

        if (Lconstraint == -2) then
          if (ihybrj <= -1) then
            Xdof(1) = dtflux(lvol) + xoffset
          elseif (ihybrj >= 0) then
            dtflux(lvol) = Xdof(1) - xoffset
          end if
        else
          select case ( ihybrj )
          case ( 0: )
            mu(lvol) = Xdof(1) - xoffset
            dpflux(lvol) = Xdof(2) - xoffset
          case ( :-1)
            ! mu and dpflux have been updated in mp00ac; early termination;
            Xdof(1) = mu(lvol) + xoffset
            Xdof(2) = dpflux(lvol) + xoffset
          end select
        end if

      else ! Lvacuumregion
        select case (ihybrj)
        case (0:)
          dtflux(lvol) = Xdof(1) - xoffset
          dpflux(lvol) = Xdof(2) - xoffset
        case (:-1)
          Xdof(1) = dtflux(lvol) + xoffset ! dtflux and dpflux have been updated in solve_beltrami_system: early termination
          Xdof(2) = dpflux(lvol) + xoffset ! dtflux and dpflux have been updated in solve_beltrami_system: early termination
        end select
      end if

      if (Lconstraint /= 2) helicity(lvol) = lABintegral(lvol)! this was computed in solve_beltrami_system

      if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lvacuumregion .and. Lconstraint == 0) .or. Lconstraint == -2) then
        iflag = 2
        Ldfjac = Ndof ! call solve_beltrami_system: compute_rotational_transform/compute_plasma_current to ensure the derivatives of B, transform, currents, wrt mu/dtflux
        ! dpflux are calculated

        call solve_beltrami_system(Ndof, Xdof(1:Ndof), Fdof(1:Ndof), Ddof(1:Ldfjac, 1:Ndof), Ldfjac, iflag)
      end if
    end select

    cput = MPI_Wtime()

    select case (ihybrj) ! this screen output may not be correct for Lvacuumregion
    case (1)
      if (Wconstruct_beltrami_field) write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), &
          cput - &
          lastcpu, "success", Fdof(1:Ndof)

    case (-2)
      if (Wconstruct_beltrami_field) write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), &
          cput - &
          lastcpu, "|F| < mupftol", Fdof(1:Ndof)

    case (-1)
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "Beltrami fail", Fdof(1:Ndof)

    case (0)
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "input error", Fdof(1:Ndof)

    case (2)
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "consider restart", Fdof(1:Ndof)

    case (3)
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "xtol too small", Fdof(1:Ndof)

    case (4:5)
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "bad  progress", Fdof(1:Ndof)

    case default
      write(stdout, 1040)cput - cpus, myid, lvol, ihybrj, helicity(lvol), mu(lvol), dpflux(lvol), cput - lastcpu, &
          "illegal ifail", Fdof(1:Ndof)

      call fatal(.true., __FILE__, "illegal ifail returned by hybrj ")
    end select

    1040 format("construct_beltrami_field : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; Linear : ihybrj =", i3, " hel="es12.4&
        " mu="es12.4" &
        &      dpflux="es12.4" time="f9.1"  ":, a16"  F="2es08.0)
  end subroutine construct_beltrami_field
end module beltrami_field_mod
