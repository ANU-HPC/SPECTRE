module profiles_distribution_mod
  use numerical_mod, only:
  use input_list_mod, only: Wmacros, Wbroadcast_profile_data, Wcompute_plasma_current, Nvol, mu, Lrad, curtor, curpol, Lconstraint,&
      & helicity, &
      enforce_stell_sym
  use allglobal_mod, only: ncpu, dtflux, dpflux, mn, Mvol, diotadxup, dItGpdxtp, Bemn, Bomn, Iomn, Iemn, Somn, Semn, &
      Pomn, Pemn, ImagneticOK, force_ft_trunc_grad_alloc, LGdof, dFFdRZ, dBBdmp, dmupfdx, denergydrr, denergydzr, &
      lBBintegral, lABintegral, vvolume, LocalConstraint, IsMyVolume, Lhessian2Dallocated
  use errors_mod, only: fatal
  use mpi_collectives_mod, only: broadcast
  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: broadcast_profile_data

contains

  subroutine broadcast_profile_data(lvol)
    integer, intent(in) :: lvol

    integer :: llmodnp, Nbc

    ! recall this routine is inside do vvol = 1, Mvol loop

    call fatal(lvol <= 0 .or. lvol > Mvol, __FILE__, "error ")

    llmodnp = modulo(lvol - 1, ncpu)! identify which node contains data: this must be consistent with previous looping /
    ! parallelization

    call broadcast(mu(lvol), llmodnp)
    call broadcast(dtflux(lvol), llmodnp)
    call broadcast(dpflux(lvol), llmodnp)
    call broadcast(helicity(lvol), llmodnp)

    call broadcast(vvolume(lvol), llmodnp)
    call broadcast(lBBintegral(lvol), llmodnp)
    call broadcast(lABintegral(lvol), llmodnp)

    call broadcast(diotadxup(0:1, -1:2, lvol), 8, llmodnp)
    call broadcast(dItGpdxtp(0:1, -1:2, lvol), 8, llmodnp)

    if (force_ft_trunc_grad_alloc) then

      if (LocalConstraint) then
        Nbc = LGdof * 2 * LGdof * 2
        call broadcast(dFFdRZ(1:LGdof, 0:1, 1:LGdof, 0:1, lvol), Nbc, llmodnp)

        Nbc = LGdof * 2 * 2
        call broadcast(dBBdmp(1:LGdof, lvol, 0:1, 1:2), Nbc, llmodnp)

        Nbc = 2 * LGdof * 2
        call broadcast(dmupfdx(lvol, 1:1, 1:2, 1:LGdof, 0:1), Nbc, llmodnp)! why is this broadcast
      end if

    end if

    if (Lhessian2Dallocated) then
      Nbc = LGdof * 2 * LGdof * 2
      call broadcast(denergydrr(1:LGdof, lvol, 0:1, 1:LGdof, 0:1), Nbc, llmodnp)
      call broadcast(denergydzr(1:LGdof, lvol, 0:1, 1:LGdof, 0:1), Nbc, llmodnp)
    end if

    call broadcast(ImagneticOK(lvol), llmodnp)
    call broadcast(Bemn(1:mn, lvol, 0:1), 2 * mn, llmodnp)! perhaps all these should be re-ordered
    call broadcast(Iomn(1:mn, lvol), mn, llmodnp)
    call broadcast(Somn(1:mn, lvol, 0:1), 2 * mn, llmodnp)
    call broadcast(Pomn(1:mn, lvol, 0:2), 3 * mn, llmodnp)

    if (.not.enforce_stell_sym) then
      call broadcast(Bomn(1:mn, lvol, 0:1), 2 * mn, llmodnp)
      call broadcast(Iemn(1:mn, lvol), mn, llmodnp)
      call broadcast(Semn(1:mn, lvol, 0:1), 2 * mn, llmodnp)
      call broadcast(Pemn(1:mn, lvol, 0:2), 3 * mn, llmodnp)
    end if

    if ((lvol > Nvol .and. Wcompute_plasma_current) .or. (Lconstraint == -2)) then
      call broadcast(curtor, llmodnp)
      call broadcast(curpol, llmodnp)
    end if

  end subroutine broadcast_profile_data
end module profiles_distribution_mod
