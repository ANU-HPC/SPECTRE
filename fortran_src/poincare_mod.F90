! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module poincare_mod
#include "mpi_precomp.h"
  USE_MPI
  use omp_lib

  use precisions_mod, only: rkind, vsmall, small
  use constants_mod, only: pi, pi2
  use numerical_mod, only:
  use io_mod, only: ounit
  use input_list_mod, only: Wmacros, &
      Wconstruct_poincare_plot, &
      Wpp00ab, &
      Nvol, &
      Lrad, &
      odetol, &
      nPpts, &
      Ppts, &
      nPtrj, &
      Lconstraint, &
      iota, &
      oita, &
      Igeometry, mpol, enforce_stell_sym
  use allglobal_mod, only: myid, ncpu, cpus, MPI_COMM_SPEC, Nz, Mvol, Lcoordinatesingularity, diotadxup, Lplasmaregion, &
      Lvacuumregion, pi2nfp, ivol, Node, in, im, mn, Ate, Aze, Ato, Azo
  use sphdf5_mod, only: init_flt_output, write_poincare, write_transform, finalize_flt_output
  use memory_allocation_mod, only: sallocate, nallocate
  use base_functions_mod, only: get_cheby, get_zernike
  use volume_mod, only: lregion
  use errors_mod, only: fatal
  use coordinate_mod, only: get_position_vetor_x

  implicit none(type, external)

  private
  public :: construct_poincare_plot

contains

  subroutine construct_poincare_plot()
    integer :: ierr
    real(rkind) :: cput

    integer :: lnPtrj, ioff, vvol, itrj, lvol
    integer, allocatable :: utflag(:), numTrajs(:)
    real(rkind) :: sti(1:2)
    real(rkind), allocatable :: data(:, :, :, :), fiota(:, :)

    integer :: recvId
    MPI_STATUS_TYPE :: status

    ! count how many Poincare trajectories should be computed in total: executed on each CPU
    call nallocate(numTrajs, 1, Mvol)
    do vvol = 1, Mvol
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      if (Lcoordinatesingularity) then
        ioff = 1 ! keep away from coordinate axis
      else
        ioff = 0
      end if

      if (nPtrj(vvol) >= 0) then
        lnPtrj = nPtrj(vvol)! selected Poincare resolution
      else
        lnPtrj = 2 * Lrad(vvol)! adapted  Poincare resolution
      end if

      numTrajs(vvol) = lnPtrj - ioff + 1 ! interval includes edge
    end do

    ! the total number of Poincare trajectories to be saved later on is given by sum(numTrajs)
    call init_flt_output(sum(numTrajs))

    ! start tracing
    do vvol = 1, Mvol

      ! skip loop content if in the wrong CPU
      if (myid == modulo(vvol - 1, ncpu) .and. myid < Mvol) then ! the following is in parallel

        ! lower bound for radial indices
        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
        if (Lcoordinatesingularity) then
          ioff = 1 ! keep away from coordinate axis
        else
          ioff = 0
        end if

        ! upper bound for radial indices
        if (nPtrj(vvol) >= 0) then
          lnPtrj = nPtrj(vvol)! selected Poincare resolution
        else
          lnPtrj = 2 * Lrad(vvol)! adapted  Poincare resolution
        end if

        call sallocate(data, ioff, lnPtrj, 1, 4, 0, Nz - 1, 1, nPpts, 0.0_rkind)! for block writing to file (allows
        ! faster reading of output data files for post-processing plotting routines)
        call sallocate(utflag, ioff, lnPtrj, 0)! error flag that indicates if fieldlines successfully followed
        call sallocate(fiota, ioff, lnPtrj, 1, 2, 0.0_rkind)! will always need fiota(0,1:2)

        !$OMP PARALLEL DO                                                                                                          &
        !$OMP SHARED(lnPtrj,ioff,Wconstruct_poincare_plot,Nz,data,fiota,utflag,iota,oita,myid,vvol,cpus,Lconstraint,nPpts,ppts)         &
        !$OMP& PRIVATE(itrj,sti) SCHEDULE(dynamic)
        do itrj = ioff, lnPtrj ! initialize Poincare plot with trajectories regularly spaced between interfaces along \t=0

          sti(1:2) = [-1.0_rkind + itrj * 2.0_rkind / lnPtrj, Ppts * pi]

          if (itrj == 0) sti(1) = -1.0_rkind ! avoid machine precision errors
          if (itrj == lnPtrj) sti(1) = 1.0_rkind ! avoid machine precision errors

          ! call actual field line integration subroutine
          call compute_poincare(vvol, sti(1:2), Nz, nPpts, data(itrj, 1:4, 0:Nz - 1, 1:nPpts), fiota(itrj, 1:2), utflag(itrj))

          if (Wconstruct_poincare_plot) then
            cput = MPI_Wtime()
            if (Lconstraint == 1) then
              if (itrj == 0) write(ounit, 1002)cput - cpus, myid, vvol, itrj, sti(1:2), utflag(itrj), fiota(itrj, 2),&
                  & fiota(itrj, 2) - oita(vvol - 1)
              if (itrj > 0 .and. itrj < lnPtrj) write(ounit, 1002)cput - cpus, myid, vvol, itrj, sti(1:2), utflag(itrj)&
                  & , fiota(itrj, 2)
              if (itrj == lnPtrj) write(ounit, 1002)cput - cpus, myid, vvol, itrj, sti(1:2), utflag(itrj), fiota(itrj,&
                  & 2), fiota(itrj, 2) - iota(vvol)
            else
              write(ounit, 1002)cput - cpus, myid, vvol, itrj, sti(1:2), utflag(itrj), fiota(itrj, 2)
            end if
          end if ! Wconstruct_poincare_plot

        end do ! itrj = ioff, lnPtrj
        !$OMP END PARALLEL DO

        ! TODO: replace below logic with a single call to MPI_allgather into rank-0
        ! and the write from there at once

        ! write data
        if (myid == 0) then
          ! The Poincare trajectories are dumped into the file one after another since this makes it easier to combine
          ! the results
          ! from all the nested volumes at the moment. Feel free to change the code, but verfiy that it works afterwards
          ! as well
          do itrj = ioff, lnPtrj
            ! write utflag --> success flag vector for field line tracing
            ! write data --> actual Poincare data
            if (vvol > 1) then
              call write_poincare(sum(numTrajs(1:vvol - 1)) + itrj - ioff, data(itrj, :, :, :), utflag)
            else
              call write_poincare(itrj - ioff, data(itrj, :, :, :), utflag)
            end if
          end do

          ! The rotational transform data is written at once for a volume
          if (vvol > 1) then
            call write_transform(sum(numTrajs(1:vvol - 1)), numTrajs(vvol), vvol, diotadxup(0:1, 0, vvol), fiota(0:lnPtrj&
                & , 1:2))
          else
            call write_transform(0, numTrajs(1), 1, diotadxup(0:1, 0, 1), fiota(ioff:lnPtrj, 1:2))
          end if

          if (Mvol > 1 .and. ncpu > 1) then

            ! Gather data from all other parallelly running CPUs: there are min(ncpu-1, Mvol-vvol) of these in this
            ! iteration
            ! If we have so few CPUs that all of them need to perform multiple iteration over the set of volumes in
            ! batches of ncpu,
            ! there are a number ncpu-1 CPUs apart from the master who still have data that needs to be written before
            ! they can contine.
            ! In the last iteration of the batch processing, or if we have many CPUs but just not enough to cover all
            ! volumes in one batch,
            ! there are Mvol-vvol (where vvol is the value of vvol in CPU 0) other CPUs waiting to have their data
            ! written.
            do recvId = 1, min(ncpu - 1, Mvol - vvol)

              ! lvol is the volume that the parallel CPU was working on
              lvol = vvol + recvId


              deallocate(utflag)
              deallocate(data)
              deallocate(fiota)

              call nallocate(utflag, 1, numTrajs(lvol))
              call nallocate(data, 1, numTrajs(lvol), 1, 4, 0, Nz - 1, 1, nPpts)
              call nallocate(fiota, 1, numTrajs(lvol), 1, 2)

              call MPI_Recv(utflag, numTrajs(lvol), MPI_integer, recvId, lvol, MPI_COMM_SPEC, status, ierr)

              call MPI_Recv(data, numTrajs(lvol) * 4 * Nz * nPpts, MPI_DOUBLE_PRECISION, recvId, lvol, MPI_COMM_SPEC,&
                  & status, ierr)

              call MPI_Recv(fiota, numTrajs(lvol) * 2, MPI_DOUBLE_PRECISION, recvId, lvol, MPI_COMM_SPEC, status, ierr)

              ! write utflag vector of CPU id
              ! write data of CPU id
              do itrj = 1, numTrajs(lvol)
                call write_poincare(sum(numTrajs(1:lvol - 1)) + itrj - 1, data(itrj, :, :, :), utflag)
              end do

              ! write fiota --> iota from field line tracing
              ! write diotadxup --> radial derivative of iota from Beltrami field (?)
              call write_transform(sum(numTrajs(1:lvol - 1)), numTrajs(lvol), lvol, diotadxup(0:1, 0, lvol),&
                  & fiota(1:numTrajs(lvol), 1:2))
            end do ! vvol = 2, Mvol
          end if

        else
          call MPI_Send(utflag, numTrajs(vvol), MPI_integer, 0, vvol, MPI_COMM_SPEC, ierr)! success flag vector
          call MPI_Send(data, numTrajs(vvol) * 4 * Nz * nPpts, MPI_DOUBLE_PRECISION, 0, vvol, MPI_COMM_SPEC, ierr)!
          !Poincare
          ! data
          call MPI_Send(fiota, numTrajs(vvol) * 2, MPI_DOUBLE_PRECISION, 0, vvol, MPI_COMM_SPEC, ierr)! rotational
          ! transform profile from field line tracing
          ! diotadxup should be available in the master already, since it is stored in global
        end if

        deallocate(data)
        deallocate(utflag)
        deallocate(fiota)

      end if
    end do ! vvol = 1, Mvol

    ! keep this until the end because we need it if we write with less CPUs than nested volumes
    deallocate(numTrajs)

    if (myid == 0) then
      call finalize_flt_output
    end if

    1002 format("construct_poincare_plot : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; ", i3, " : (s,t)=(", f21.17, " ,", &
        f21.17, " ) ;", :, " utflag=", i3, "  transform=", es23.15, " ;", :, " error=", es13.5, " ")
  end subroutine construct_poincare_plot

  subroutine compute_poincare(lvol, sti, Nz, nPpts, poincaredata, fittedtransform, utflag)
    real(rkind) :: cput

    integer, intent(in) :: lvol, Nz, nPpts
    integer, intent(out) :: utflag
    real(rkind) :: sti(1:2), poincaredata(1:4, 0:Nz - 1, 1:nPpts), fittedtransform(1:2), dzeta

    integer :: jj, kk
    real(rkind) :: ppt(1:4)

    integer, parameter :: Lrwork = 20 * Node
    real(rkind) :: zst, zend, st(1:Node), tol, stz(1:3), RpZ(1:3), leastfit(1:5)
    character :: RA

    integer, parameter :: Lenwrk = 32 * Node
    integer :: rkmethod, outch
    real(rkind) :: hstart, thres(1:Node), rkwork(1:Lenwrk), mchpes, dwarf
    real(rkind) :: zgot, ygot(1:Node), ypgot(1:Node), ymax(1:Node)
    character :: rktask
    logical :: errass, mesage

    external :: SETUP, UT, ENVIRN

    ivol = lvol ! required to pass through to compute_magnetic_field

    dzeta = pi2nfp

    RA = 'D'
    tol = odetol

    st(1:Node) = [sti(1), sti(2)] ! , 1.0_rkind, 0.0_rkind, 0.0_rkind, 1.0_rkind /)!, 0.0_rkind, 0.0_rkind /)

    ppt(1:2) = [sti(2), sti(1)] ! seems redundant: just used for packing into poincaredata it seems

    leastfit(1:5) = [0.0_rkind, 0.0_rkind, 0.0_rkind, ppt(1), 1.0_rkind] ! initialize summation for least squares fit

    utflag = 0
    poincaredata(1:4, 0:Nz - 1, 1:nPpts) = 0.0_rkind
    fittedtransform(1:2) = -2.0_rkind ! provide dummy defaults

    do jj = 1, nPpts ! loop over iterations

      zst = 0.0_rkind ! starting Poincare plane

      call ENVIRN(outch, mchpes, dwarf)! only dwarf is used to set thres=sqrt(dwarf): thres could be set larger but not
      ! smaller
      thres(1:Node) = sqrt(dwarf)
      rkmethod = 3
      rktask = 'U'
      errass = .false.
      hstart = 0.0e0_rkind

      mesage = .false.

      do kk = 0, Nz - 1 ! loop over toroidal Poincare cross sections

        stz(1:3) = [ppt(2), mod(ppt(1), pi2), zst] ! toroidal coordinates

        if (abs(stz(1)) > 1.0_rkind) then

          utflag = 0
          exit ! exit do kk loop

        end if

        call get_position_vetor_x(lvol, stz(1:3), RpZ(1:3))! map to cylindrical

        ppt(3:4) = [RpZ(1), RpZ(3)] ! cylindrical coordinates

        poincaredata(1:4, kk, jj) = [mod(ppt(1), pi2), ppt(2), ppt(3), ppt(4)] ! save Poincare information to array

        zend = zst + (pi2nfp / Nz)

        utflag = 0

        call SETUP(Node, zst, st(1:Node), zend, tol, thres(1:Node), rkmethod, rktask, errass, hstart, rkwork(1:Lenwrk),&
            & Lenwrk, mesage)

        ! integrate to next plane
        call UT(pnc_compute_magnetic_field, zend, zgot, ygot(1:Node), ypgot(1:Node), ymax(1:Node), rkwork(1:Lenwrk), utflag)

        zst = zend

        st(1:Node) = ygot(1:Node)

        cput = MPI_Wtime()
        select case (utflag)
        case (1)
          ! give screen output if error is encountered
        case (2)
          write(ounit, 2001)cput - cpus, myid, lvol, jj, kk, utflag, "step size too small (try RK method 2)"
        case (3)
          write(ounit, 2001)cput - cpus, myid, lvol, jj, kk, utflag, "integration interrupted (more than 5000 calls)"
        case (4)
          write(ounit, 2001)cput - cpus, myid, lvol, jj, kk, utflag, "problem is stiff (UT is not efficient)"
        case (5)
          write(ounit, 2001)cput - cpus, myid, lvol, jj, kk, utflag, "odetol/thres too small or RK method too low"
        case (6)
          write(ounit, 2001)cput - cpus, myid, lvol, jj, kk, utflag, "integration interrupted (error assessment not    &
              &         possible"
        case default
          call fatal(.true., __FILE__, "illegal value of ifail returned from UT")

        end select

        2001 format("pp00ab : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (jj,kk)=(", i4, " ,", i4, " ); ifail=", i2, "  ", a63)

        if (utflag /= 1) exit ! an integration error was encountered: exit do kk loop

        ppt(1:2) = [st(2), st(1)] ! again, seems redundant: I think this is only required for packing into poincaredata

      end do

      if (utflag /= 1) exit ! an integration error was encountered: exit do jj loop

      leastfit(1:5) = leastfit(1:5) + [ (jj * dzeta) ** 2, (jj * dzeta), (jj * dzeta) * ppt(1), ppt(1), 1.0_rkind] ! least
      ! squares
      ! fit summation

    end do

    if (utflag == 1) then
      fittedtransform(1:2) = [sti(1), (leastfit(5) * leastfit(3) - leastfit(2) * leastfit(4)) / (leastfit(5) *&
          & leastfit(1) - leastfit(2) * leastfit(2)) ]
    end if
  end subroutine compute_poincare

  subroutine pnc_compute_magnetic_field(zeta, st, Bst)

    real(rkind), intent(in) :: zeta, st(1:Node)
    real(rkind), intent(out) :: Bst(1:Node)
    integer :: lvol

    real(rkind) :: cput
    integer :: ii, ll, mi, ni, ideriv
    real(rkind) :: teta, lss, sbar, arg, carg, sarg, dBu(1:3)

    real(rkind) :: cheby(0:Lrad(ivol), 0:1), zernike(0:Lrad(1), 0:Mpol, 0:1)

    real(rkind) :: TT(0:Lrad(ivol), 0:1)! this is almost identical to cheby

    lvol = ivol

    ideriv = 0 ! the argument list of compute_magnetic_field is fixed by NAG requirements, but volume index is required below

    Bst(1:Node) = [0.0_rkind, 0.0_rkind] ! set default intent out: this should cause a compilation error if Node.ne.2

    lss = st(1)
    teta = st(2)

    if (abs(lss) <= 1.0_rkind) then

      if (Lcoordinatesingularity) sbar = max( (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind, small)

      if (Lcoordinatesingularity) then
        call get_zernike(sbar, Lrad(lvol), Mpol, zernike(:, :, 0:1))
      else
        call get_cheby(lss, Lrad(lvol), cheby(0:Lrad(lvol), 0:1))
      end if

      dBu(1:3) = 0.0_rkind ! initialize summation

      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)
        arg = mi * teta - ni * zeta
        carg = cos(arg)
        sarg = sin(arg)! shorthand

        if (Lcoordinatesingularity) then ! regularization factor depends on mi

          call fatal(abs(sbar) < vsmall, __FILE__, "need to avoid divide-by-0.0_rkind ")

          do ll = 0, Lrad(lvol)
            TT(ll, 0:1) = [zernike(ll, mi, 0), zernike(ll, mi, 1) * 1.0_rkind / 2.0_rkind]
          end do

        else

          do ll = 0, Lrad(lvol)
            TT(ll, 0:1) = [cheby(ll, 0), cheby(ll, 1)]
          end do

        end if

        do ll = 0, Lrad(lvol)! loop over Chebyshev summation
          dBu(1) = dBu(1) + (-mi * Aze(lvol, ideriv, ii)%s(ll) - ni * Ate(lvol, ideriv, ii)%s(ll)) * TT(ll, 0) * sarg
          dBu(2) = dBu(2) + (-Aze(lvol, ideriv, ii)%s(ll)) * TT(ll, 1) * carg
          dBu(3) = dBu(3) + (Ate(lvol, ideriv, ii)%s(ll)) * TT(ll, 1) * carg
          if (.not.enforce_stell_sym) then ! include non-symmetric harmonics
            dBu(1) = dBu(1) + (+mi * Azo(lvol, ideriv, ii)%s(ll) + ni * Ato(lvol, ideriv, ii)%s(ll)) * TT(ll, 0) * carg
            dBu(2) = dBu(2) + (-Azo(lvol, ideriv, ii)%s(ll)) * TT(ll, 1) * sarg
            dBu(3) = dBu(3) + (Ato(lvol, ideriv, ii)%s(ll)) * TT(ll, 1) * sarg
          end if
        end do

      end do

      if (abs(dBu(3)) < vsmall) then

        cput = MPI_Wtime()

        write(ounit, '("compute_magnetic_field : ",f10.2," : lvol=",i3," ; zeta="es23.15" ; (s,t)=("es23.15" ,"es23.15" ) ; B^z         &
            & ="es23.15" ")')cput - cpus, lvol, zeta, st(1:2), dBu(3)

        call fatal(abs(dBu(3)) < vsmall, __FILE__, "field is not toroidal ")
      end if

      Bst(1:2) = dBu(1:2) / dBu(3)! normalize field line equations to toroidal field
    end if
  end subroutine pnc_compute_magnetic_field

end module poincare_mod
