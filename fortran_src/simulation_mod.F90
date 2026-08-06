! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

module simulation_mod
#include "mpi_precomp.h"
  USE_MPI

  use allglobal_mod, only: cpus, myid, ncpu, Mvol, Ntz, mn, beltramierror, Lcoordinatesingularity, Lplasmaregion, &
      Lvacuumregion, Btemn, Bzemn, Btomn, Bzomn, IPDt, ImagneticOK, dtflux, Iquad, lmns, Nt, Nz, diotadxup, IsMyVolume, &
      IsMyVolumeValue, WhichCpuID, dlambdaout
  use constants_mod, only: pi2
  use diagnostics_mod, only: compute_beltrami_error
  use errors_mod, only: fatal
  use input_list_mod, only: nPtrj, nPpts, Igeometry, Lcheck, Nvol, odetol, Ivolume, mu, Ltransform, Lsvdiota
  use io_mod, only: ounit
  use magnetic_field_mod, only: compute_rotational_transform
  use mpi_collectives_mod, only: broadcast
  use fft_backend_mod, only: fft_init
  use plasma_current_mod, only: interface_bfield_harmonics
  use poincare_mod, only: construct_poincare_plot
  use precisions_mod, only: rkind
  use profiles_distribution_mod, only: broadcast_profile_data
  use volume_mod, only: lregion
  use field_mod, only: alloc_ntz_arrays

  implicit none

  private
  public :: final_diagnostics

contains

  subroutine final_diagnostics
    real(rkind) :: cput

    integer :: iocons, llmodnp, vvol, iflag, cpu_id
    real :: sumI

    call fft_init(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    ! Evaluate rotational transform and straight field line coordinate transformation
    if (Ltransform) then

      call fatal(Lsvdiota /= 1, __FILE__, "Lsvdiota needs to be 1.0_rkind for s.f.l transformation")

      do vvol = 1, Mvol
        call broadcast_profile_data(vvol)
      end do

      iflag = -1
      do vvol = 1, Mvol
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) then
          cycle
        elseif (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

        call compute_rotational_transform(vvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, vvol), -1)! stores lambda in a global variable.
      end do

      ! Broadcast
      do vvol = 1, Mvol
        cpu_id = WhichCpuID(vvol, ncpu)
        call broadcast(diotadxup(0:1, -1:2, vvol), 8, cpu_id)
        call broadcast(dlambdaout(1:lmns, vvol, 0:1), 2 * lmns, cpu_id)
      end do

    end if

    ! Computes the surface current at each interface for output

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      do iocons = 0, 1
        if (Lcoordinatesingularity .and. iocons == 0) cycle
        if (vvol == Nvol + 1 .and. iocons == 1) cycle
        ! Compute covariant magnetic field at interface, directly into the output arrays
        call interface_bfield_harmonics(vvol, Nt, Nz, Lcoordinatesingularity, 0, iocons, -1, &
            Btemn(1:mn, iocons, vvol), Btomn(1:mn, iocons, vvol), Bzemn(1:mn, iocons, vvol), Bzomn(1:mn, iocons, vvol))
      end do
    end do

    ! Evaluate surface current from the jump of the (m,n)=(0,0) covariant B_theta across each interface
    do vvol = 1, Mvol - 1
      IPDt(vvol) = pi2 * (Btemn(1, 0, vvol + 1) - Btemn(1, 1, vvol))
    end do

    ! Evaluate volume current
    sumI = 0
    do vvol = 1, Mvol
      Ivolume(vvol) = mu(vvol) * dtflux(vvol) * pi2 + sumI ! factor pi2 due to normalization in initialize_internal_arrays
      sumI = Ivolume(vvol)! Sum over all volumes since this is how Ivolume is defined
    end do

    ! screen info about diagnostics
    if (myid == 0) then
      cput = MPI_Wtime()

      if (nPpts > 0) then
        write(ounit, '("xspech : ", 10x ," :")')
	write(ounit, '("xspech : ",f10.2," : myid=",i3," ; Poincare plot ; odetol=",es8.1,' // &
              '" ; nPpts=",i7," ",:," nPtrj=",24(i5," "))') &
      	cput - cpus, myid, odetol, nPpts, nPtrj(1:min(Mvol, 24))
      end if

      if (Lcheck == 1) then
        write(ounit, '("xspech : ", 10x ," :")')
        write(ounit, '("xspech : ",f10.2," : myid=",i3,"  calling compute_beltrami_error computing error in field ")')cput - cpus,&
            & myid
      end if
    end if

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      if (myid == modulo(vvol - 1, ncpu) .and. myid < Mvol) then ! the following is in parallel

        if (.not.ImagneticOK(vvol)) then
          cput = MPI_Wtime()
          write(ounit, 1002)cput - cpus
          write(ounit, 1002)cput - cpus, myid, vvol, ImagneticOK(vvol)
          cycle
        end if

        if (Lcheck == 1) then
          call compute_beltrami_error(vvol, Ntz, Iquad(vvol), mn)
        end if

      end if
    end do

    if (nPpts > 0) then
      call construct_poincare_plot()! do Poincare plots in all volumes: has its own paralellization over volumes internally
    end if

    1002 format("xspech : ", f10.2, " : myid=", i3, " ; vvol=", i3, " ; IBeltrami=", L2, " construction of Beltrami field  failed ")

    do vvol = 1, Mvol
      llmodnp = modulo(vvol - 1, ncpu)
      call broadcast(Btemn(1:mn, 0:1, vvol), mn * 2, llmodnp)! this is computed in interface_bfield_harmonics
      call broadcast(Bzemn(1:mn, 0:1, vvol), mn * 2, llmodnp)
      call broadcast(Btomn(1:mn, 0:1, vvol), mn * 2, llmodnp)
      call broadcast(Bzomn(1:mn, 0:1, vvol), mn * 2, llmodnp)

      call broadcast(beltramierror(vvol, 1:9), 9, llmodnp)! this is computed in compute_beltrami_error

    end do

  end subroutine final_diagnostics
end module simulation_mod
