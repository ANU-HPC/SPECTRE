! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module magnetic_field_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind, vsmall, small
  use numerical_mod, only: invfft
  use io_mod, only: ounit
  use input_list_mod, only: Mpol, enforce_stell_sym, Lrad, &
      Wcompute_rotational_transform, Ntor, Lsparse, Lsvdiota
  use allglobal_mod, only: cpus, mn, im, in, Ate, Aze, Ato, Azo, Lcoordinatesingularity, Node, &
      ncpu, myid, &
      TT, mns, ims, ins, &
      iotakkii, iotaksub, iotakadd, iotaksgn, RTT, Lvacuumregion, dlambdaout, ivol
  use errors_mod, only: fatal
  use base_functions_mod, only: get_cheby, get_zernike
  use constants_mod, only: pi2, pi
  use memory_allocation_mod, only: sallocate

  implicit none(type, external)

  private
  public :: compute_magnetic_field, compute_gB_contravariant, compute_rotational_transform

  external :: DCOPY, DGEMV, dgesvx, dgelsd

contains

  !> (sqrt(g) B^s, sqrt(g) B^theta, sqrt(g) B^zeta) at (s, theta, zeta) from vector-potential
  !> slot aslot: 0 = the field, -1 = the geometry-perturbed solution written by
  !> get_perturbed_solution. These combinations are metric-free in the coefficients, so slot -1
  !> alone is the complete derivative of sqrt(g) B at fixed (s, theta, zeta) and fixed fluxes.
  subroutine compute_gB_contravariant(lvol, zeta, st, aslot, gB)
    integer, intent(in) :: lvol, aslot
    real(rkind), intent(in) :: zeta, st(1:Node)
    real(rkind), intent(out) :: gB(1:3)

    integer :: ii, ll, mi, ni
    real(rkind) :: teta, lss, sbar, arg, carg, sarg

    real(rkind) :: cheby(0:Lrad(lvol), 0:1), zernike(0:Lrad(1), 0:Mpol, 0:1)

    real(rkind) :: TT(0:Lrad(lvol), 0:1)! this is almost identical to cheby

    gB(1:3) = 0.0_rkind

    lss = st(1)
    teta = st(2)

    if (abs(lss) <= 1.0_rkind) then

      if (Lcoordinatesingularity) sbar = max( (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind, small)

      if (Lcoordinatesingularity) then
        call get_zernike(sbar, Lrad(lvol), Mpol, zernike(:, :, 0:1))
      else
        call get_cheby(lss, Lrad(lvol), cheby(0:Lrad(lvol), 0:1))
      end if

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
          gB(1) = gB(1) + (-mi * Aze(lvol, aslot, ii)%s(ll) - ni * Ate(lvol, aslot, ii)%s(ll)) * TT(ll, 0) * sarg
          gB(2) = gB(2) + (-Aze(lvol, aslot, ii)%s(ll)) * TT(ll, 1) * carg
          gB(3) = gB(3) + (Ate(lvol, aslot, ii)%s(ll)) * TT(ll, 1) * carg
          if (.not.enforce_stell_sym) then ! include non-symmetric harmonics
            gB(1) = gB(1) + (+mi * Azo(lvol, aslot, ii)%s(ll) + ni * Ato(lvol, aslot, ii)%s(ll)) * TT(ll, 0) * carg
            gB(2) = gB(2) + (-Azo(lvol, aslot, ii)%s(ll)) * TT(ll, 1) * sarg
            gB(3) = gB(3) + (Ato(lvol, aslot, ii)%s(ll)) * TT(ll, 1) * sarg
          end if
        end do

      end do

    end if
  end subroutine compute_gB_contravariant

  subroutine compute_magnetic_field(lvol, zeta, st, Bst, gBzeta)
    integer, intent(in) :: lvol
    real(rkind), intent(in) :: zeta, st(1:Node)
    real(rkind), intent(out) :: Bst(1:Node), gBzeta

    real(rkind) :: cput
    real(rkind) :: dBu(1:3)

    Bst(1:Node) = [0.0_rkind, 0.0_rkind] ! set default intent out: this should cause a compilation error if Node.ne.2

    if (abs(st(1)) <= 1.0_rkind) then

      call compute_gB_contravariant(lvol, zeta, st, 0, dBu)

      gBzeta = dBu(3)! NOPE: gBzeta is returned through global

      if (abs(gBzeta) < vsmall) then

        cput = MPI_Wtime()

        write(ounit, '("compute_magnetic_field : ",f10.2," : lvol=",i3,' // &
            '" ; zeta=",es23.15," ; (s,t)=(",es23.15,",",es23.15,") ; B^z =",es23.15," ")') &
            cput - cpus, lvol, zeta, st(1:2), dBu(3)

        call fatal(abs(dBu(3)) < vsmall, __FILE__, "field is not toroidal ")
      end if

      Bst(1:2) = dBu(1:2) / gBzeta ! normalize field line equations to toroidal field
    end if
  end subroutine compute_magnetic_field

  subroutine compute_rotational_transform(lvol, mn, NN, Nt, Nz, Lcoordinatesingularity, iflag, ldiota, aslot)! construct straight-field line magnetic coordinates
    real(rkind) :: cput

    integer, intent(in) :: lvol, mn, NN, Nt, Nz, iflag
    logical, intent(in) :: Lcoordinatesingularity

    real(rkind), intent(inout) :: ldiota(0:1, -1:2)
    integer, intent(in) :: aslot ! perturbed Ate/... slot for the ideriv=-1 read; -1 when serial

    integer :: innout, ll, ii, jj, kk, mj, nj, ideriv, jderiv, id, MM, mi, imupf, rslot

    real(rkind) :: lcpu

    real(rkind) :: lAte(0:mn, -1:2), lAze(0:mn, -1:2), lAto(0:mn, -1:2), lAzo(0:mn, -1:2)

    ! required for Fourier routines
    integer :: if04aaf, idgesvx, ipiv(1:NN), iwork4(1:NN)
    real(rkind), allocatable :: dmatrix(:, :, :), omatrix(:, :), FAA(:, :)
    real(rkind) :: drhs(1:NN, -1:2), dlambda(1:NN, -1:2)
    real(rkind) :: Rdgesvx(1:NN), Cdgesvx(1:NN), work4(1:4 * NN), rcond, ferr(1), berr(1), ferr2(1:2), berr2(1:2)
    character :: equed

    !required for SVD routines
    integer :: idgelsd, Lwork, Liwork, Irank, nlvl
    real(rkind) :: sval(1:NN)
    real(rkind), allocatable :: work(:)

    real(rkind) :: Bsupt(1:Nt * Nz, -1:2), Bsupz(1:Nt * Nz, -1:2)

    integer, allocatable :: iwork(:)

    do innout = 0, 1 ! loop over inner and outer interfaces

      if (Lcoordinatesingularity .and. innout == 0) cycle ! transform on coordinate axis is not required
      if (Lvacuumregion .and. innout == 1) cycle ! transform on computational boundary is not required (not defined)

      lAte(0:mn, -1:2) = 0.0_rkind ! radial derivatives of vector potential evaluated at interfaces
      lAze(0:mn, -1:2) = 0.0_rkind
      !  if( .not.enforce_stell_sym ) then ! the non-stellarator-symmetric harmonics need to be set to zero in any case
      lAto(0:mn, -1:2) = 0.0_rkind
      lAzo(0:mn, -1:2) = 0.0_rkind
      !  endif

      do ideriv = -1, 2
        id = ideriv ! labels derivative of magnetic field wrt enclosed fluxes
        rslot = merge(aslot, id, id == -1) ! -1 is the per-DOF perturbed Ate storage slot; 0..2 shared

        if (iflag == 1 .and. ideriv /= 0) cycle ! derivatives of transform are not required
        if (iflag == 2 .and. ideriv < 0) cycle ! derivatives of transform wrt geometry is  not required
        if (iflag == -1 .and. ideriv > 0) cycle ! derivatives of transform wrt helicity multiplier and differential
        ! poloidal flux are not required

        do ii = 1, mn ! loop over Fourier harmonics

          mi = im(ii)

          if (Lcoordinatesingularity) then
            do ll = mi, Lrad(lvol), 2 ! loop over Zernike polynomials

              lAte(ii, id) = lAte(ii, id) + Ate(lvol, rslot, ii)%s(ll) * RTT(ll, mi, innout, 1) * 1.0_rkind / 2.0_rkind
              lAze(ii, id) = lAze(ii, id) - Aze(lvol, rslot, ii)%s(ll) * RTT(ll, mi, innout, 1) * 1.0_rkind / 2.0_rkind
              if (.not.enforce_stell_sym) then
                lAto(ii, id) = lAto(ii, id) + Ato(lvol, rslot, ii)%s(ll) * RTT(ll, mi, innout, 1) * 1.0_rkind / 2.0_rkind
                lAzo(ii, id) = lAzo(ii, id) - Azo(lvol, rslot, ii)%s(ll) * RTT(ll, mi, innout, 1) * 1.0_rkind / 2.0_rkind
              end if

            end do
          else
            do ll = 0, Lrad(lvol)! loop over Chebyshev polynomials

              lAte(ii, id) = lAte(ii, id) + Ate(lvol, rslot, ii)%s(ll) * TT(ll, innout, 1)! compute radial derivative of
              ! vector potential
              lAze(ii, id) = lAze(ii, id) - Aze(lvol, rslot, ii)%s(ll) * TT(ll, innout, 1)
              if (.not.enforce_stell_sym) then
                lAto(ii, id) = lAto(ii, id) + Ato(lvol, rslot, ii)%s(ll) * TT(ll, innout, 1)
                lAzo(ii, id) = lAzo(ii, id) - Azo(lvol, rslot, ii)%s(ll) * TT(ll, innout, 1)
              end if

            end do
          end if ! Lcoordinatesingularity
        end do

        if (Lsparse > 0) then ! will construct transformation to straight-field line angle in real space
          call invfft(mn, im(1:mn), in(1:mn), lAte(1:mn, id), lAto(1:mn, id), lAze(1:mn, id), lAzo(1:mn, id), &
              & Bsupz(1:(Nt*Nz), id), Bsupt(1:(Nt*Nz), id))! map to real space
        end if

      end do

      ! construct real-space, real-space transformation matrix

      if (Lsparse == 0 .or. Lsparse == 3) then
        call sallocate(dmatrix, 1, NN, 1, NN, -1, 2, 0.0_rkind)
        call sallocate(omatrix, 1, NN, 1, NN, 0.0_rkind)
        call sallocate(FAA, 1, NN, 1, NN, 0.0_rkind)
      end if

      if (Lsparse == 0 .or. Lsparse == 3) then ! Fourier transformation

        drhs(1:NN, -1:2) = 0.0_rkind

        dmatrix(1:NN, 1:NN, -1:2) = 0.0_rkind ! initialize summation

        do ideriv = -1, 2
          if (iflag == 1 .and. ideriv /= 0) cycle ! derivatives not required
          if (iflag == 2 .and. ideriv < 0) cycle ! derivatives wrt helicity multiplier and differential poloidal flux are required
          if (iflag == -1 .and. ideriv > 0) cycle ! derivative  wrt geometry required

          !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(kk,ii,jj,mj,nj)
          do kk = 1, mn

            ii = iotakkii(kk)

            drhs(ii, ideriv) = +lAze(kk, ideriv)
            if (.not.enforce_stell_sym .and. kk > 0) then
              drhs(ii + mns - 1, ideriv) = +lAzo(kk, ideriv)
            end if

            dmatrix(ii, 1, ideriv) = lAte(kk, ideriv)
            if (.not.enforce_stell_sym .and. kk > 0) then
              dmatrix(ii + mns - 1, 1, ideriv) = lAto(kk, ideriv)
            end if

            do jj = 2, mns
              mj = ims(jj)
              nj = ins(jj)! this seems to ignore the non-stellarator symmetric mode
              ii = iotakadd(kk, jj)
              if (ii < 1) cycle

              !$OMP ATOMIC UPDATE
              dmatrix(ii, jj, ideriv) = dmatrix(ii, jj, ideriv) + (-mj * lAze(kk, ideriv) + nj * lAte(kk, ideriv)) *&
                  & 1.0_rkind / 2.0_rkind
              if (.not.enforce_stell_sym) then
                call fatal(ii + mns - 1 < 1 .or. ii + mns - 1 > NN .or. jj + mns - 1 < 1 .or. jj + mns - 1 > NN, __FILE__&
                    & , "illegal subscript ")
                ! THIS CAN BE DELETED EVENTUALLY
                !$OMP ATOMIC UPDATE
                dmatrix(ii + mns - 1, jj, ideriv) = dmatrix(ii + mns - 1, jj, ideriv) + (-mj * lAzo(kk, ideriv) + nj *&
                    & lAto(kk, ideriv)) * 1.0_rkind / 2.0_rkind
                !$OMP ATOMIC UPDATE
                dmatrix(ii, jj + mns - 1, ideriv) = dmatrix(ii, jj + mns - 1, ideriv) - (+mj * lAzo(kk, ideriv) - nj *&
                    & lAto(kk, ideriv)) * 1.0_rkind / 2.0_rkind
                !$OMP ATOMIC UPDATE
                dmatrix(ii + mns - 1, jj + mns - 1, ideriv) = dmatrix(ii + mns - 1, jj + mns - 1, ideriv) + (+mj *&
                    & lAze(kk, ideriv) - nj * lAte(kk, ideriv)) * 1.0_rkind / 2.0_rkind
              end if

            end do

            do jj = 2, mns
              mj = ims(jj)
              nj = ins(jj)

              ii = iotaksub(kk, jj)

              if (ii < 1) cycle

              call fatal(ii > NN .or. jj > NN, __FILE__, "illegal subscript ")
              ! THIS CAN BE DELETED EVENTUALLY
              !$OMP ATOMIC UPDATE
              dmatrix(ii, jj, ideriv) = dmatrix(ii, jj, ideriv) + (-mj * lAze(kk, ideriv) + nj * lAte(kk, ideriv)) *&
                  & 1.0_rkind / 2.0_rkind
              if (.not.enforce_stell_sym) then
                call fatal(ii + mns - 1 < 1 .or. ii + mns - 1 > NN .or. jj + mns - 1 < 1 .or. jj + mns - 1 > NN, __FILE__&
                    & , "illegal subscript ")
                ! THIS CAN BE DELETED EVENTUALLY
                !$OMP ATOMIC UPDATE
                dmatrix(ii + mns - 1, jj, ideriv) = dmatrix(ii + mns - 1, jj, ideriv) + (-mj * lAzo(kk, ideriv) + nj *&
                    & lAto(kk, ideriv)) * 1.0_rkind / 2.0_rkind * iotaksgn(kk, jj)
                !$OMP ATOMIC UPDATE
                dmatrix(ii, jj + mns - 1, ideriv) = dmatrix(ii, jj + mns - 1, ideriv) + (+mj * lAzo(kk, ideriv) - nj *&
                    & lAto(kk, ideriv)) * 1.0_rkind / 2.0_rkind
                !$OMP ATOMIC UPDATE
                dmatrix(ii + mns - 1, jj + mns - 1, ideriv) = dmatrix(ii + mns - 1, jj + mns - 1, ideriv) - (+mj *&
                    & lAze(kk, ideriv) - nj * lAte(kk, ideriv)) * 1.0_rkind / 2.0_rkind * iotaksgn(kk, jj)
              end if

            end do
          end do
          !$OMP END PARALLEL DO
        end do

        ! FOURIER MATRICES HAVE BEEN CONSTRUCTED: SAVE UNPERTURBED MATRIX AND UNPERTURBED SOLUTION: FOR FUTURE USE

        !omatrix(1:NN,1:NN) = dmatrix(1:NN,1:NN,0) ! original "unperturbed" matrix
        call DCOPY(NN * NN, dmatrix(1, 1, 0), 1, omatrix(1, 1), 1)! BLAS version

        do jderiv = 0, 1

          if (iflag == 1 .and. jderiv /= 0) cycle ! derivatives of rotational transform (wrt either
          ! enclosed-fluxes/currents/geometry) are not required

          select case (jderiv)
          case (0)
          case (1)
            if (iflag == 2) then
              call DGEMV('N', NN, NN, -1.0_rkind, dmatrix(1, 1, 1), NN, dlambda(1, 0), 1, 1.0_rkind, drhs(1, 1), 1)!
              !BLAS version
              call DGEMV('N', NN, NN, -1.0_rkind, dmatrix(1, 1, 2), NN, dlambda(1, 0), 1, 1.0_rkind, drhs(1, 2), 1)!
              !BLAS version
            end if
            if (iflag == -1) then
              call DGEMV('N', NN, NN, -1.0_rkind, dmatrix(1, 1, -1), NN, dlambda(1, 0), 1, 1.0_rkind, drhs(1, -1), 1)!
              !BLAS version
            end if
          case default
            call fatal(.true., __FILE__, "invalid jderiv ")

          end select

          lcpu = MPI_Wtime()! record time taken in dgesvx

          select case (Lsvdiota)

          case (0) ! Lsvdiota = 0: use linear solver to invert linear equations that define the straight fieldline angle

            if04aaf = 1

            select case (jderiv)

            case (0) ! Lsvdiota = 0: jderiv = 0

              MM = 1

              call dgesvx('N', 'N', NN, MM, dmatrix(1:NN, 1:NN, 0), NN, FAA(1:NN, 1:NN), NN, ipiv(1:NN), equed,&
                  & Rdgesvx(1:NN), Cdgesvx(1:NN), drhs(1:NN, 0:0), NN, dlambda(1:NN, 0:0), NN, rcond, ferr, berr,&
                  & work4(1:4 * NN), iwork4(1:NN), idgesvx)

              ldiota(innout, 0) = dlambda(1, 0)! return intent out
              dlambdaout(1:NN, lvol, innout) = dlambda(1:NN, 0)

            case (1) ! Lsvdiota = 0: jderiv = 1

              MM = 2
              if (iflag == -1) then
                drhs(1:NN, 1) = drhs(1:NN, -1)
                drhs(1:NN, 2) = 0.0_rkind
              end if

              !dmatrix(1:NN,1:NN,0) = omatrix(1:NN,1:NN) ! original "unperturbed" matrix
              call DCOPY(NN * NN, omatrix(1, 1), 1, dmatrix(1, 1, 0), 1)! BLAS version

              call dgesvx('N', 'N', NN, MM, dmatrix(1:NN, 1:NN, 0), NN, FAA(1:NN, 1:NN), NN, ipiv(1:NN), equed,&
                  & Rdgesvx(1:NN), Cdgesvx(1:NN), drhs(1:NN, 1:MM), NN, dlambda(1:NN, 1:MM), NN, rcond, ferr2(1:MM),&
                  & berr2(1:MM), work4(1:4 * NN), iwork4(1:NN), idgesvx)

              if (iflag == 2) ldiota(innout, 1:2) = dlambda(1, 1:2)! return intent out
              if (iflag == -1) ldiota(innout, -1) = dlambda(1, 1)! return intent out

            case default

              call fatal(.true., __FILE__, "invalid jderiv ")

            end select

            cput = MPI_Wtime()

            select case (idgesvx) !12345678901234567
            case (0)
              if (Wcompute_rotational_transform) write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgesvx", idgesvx, cput -&
                  & lcpu,&
                  & "solved Fourier ", dlambda(1, 0)
            case (1:)
              write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgesvx", idgesvx, cput - lcpu, "singular"
            case (:-1)
              write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgesvx", idgesvx, cput - lcpu, "input error"
            case default
              call fatal(.true., __FILE__, "illegal ifail returned by dgesvx ")

            end select

            call fatal(idgesvx /= 0, __FILE__, "failed to construct straight-fieldline angle using dgesvx ")

          case (1) ! Lsvdiota = 1: use least-squares to invert linear equations that define the straight fieldline angle

            !     Here Lwork = 12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2  and Liwork = (11+3*NLVL)*NN
            !     where SMLSIZ=25 and NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 )
            !     Warning: this could become insufficient in some cases and produce input error message
            nlvl = max(0, int(log(real(NN, rkind) / 26) / log(2.0_rkind)) + 1)
            Lwork = (63 + 8 * nlvl) * NN + 676
            Liwork = max(1, 11 * NN + 3 * nlvl * NN)

            call sallocate(work, 1, Lwork, 0.0_rkind)
            if (allocated(iwork)) then
              deallocate(iwork)
            end if
            call sallocate(iwork, 1, Liwork, 0)

            select case (jderiv)

            case (0) ! Lsvdiota = 1: jderiv = 0

              dlambda(1:NN, 0) = drhs(1:NN, 0)! on entry, rhs: on exit, solution

              call dgelsd(NN, NN, 1, dmatrix(1:NN, 1:NN, 0), NN, dlambda(1:NN, 0), NN, sval(1:NN), rcond, Irank,&
                  & work(1:Lwork), Lwork, iwork(1:Liwork), idgelsd)

              ldiota(innout, 0) = dlambda(1, 0)
              dlambdaout(1:NN, lvol, innout) = dlambda(1:NN, 0)

            case (1) ! Lsvdiota = 1: jderiv = 1

              if (iflag == 2) then
                do imupf = 1, 2
                  dmatrix(1:NN, 1:NN, 0) = omatrix(1:NN, 1:NN)
                  dlambda(1:NN, imupf) = drhs(1:NN, imupf)

                  call dgelsd(NN, NN, 1, dmatrix(1:NN, 1:NN, 0), NN, dlambda(1:NN, imupf), NN, sval(1:NN), rcond, Irank,&
                      & work(1:Lwork), Lwork, iwork(1:Liwork), idgelsd)

                  ldiota(innout, imupf) = dlambda(1, imupf)
                end do
              elseif (iflag == -1) then
                do imupf = -1, -1
                  dmatrix(1:NN, 1:NN, 0) = omatrix(1:NN, 1:NN)
                  dlambda(1:NN, imupf) = drhs(1:NN, imupf)

                  call dgelsd(NN, NN, 1, dmatrix(1:NN, 1:NN, 0), NN, dlambda(1:NN, imupf), NN, sval(1:NN), rcond, Irank,&
                      & work(1:Lwork), Lwork, iwork(1:Liwork), idgelsd)

                  ldiota(innout, imupf) = dlambda(1, imupf)
                end do
              else
                call fatal(.true., __FILE__, "invalid iflag ")

              end if

            case default

              call fatal(.true., __FILE__, "invalid jderiv ")

            end select

            deallocate(work)

            cput = MPI_Wtime()

            select case (idgelsd) !12345678901234567
            case (0)
              if (Wcompute_rotational_transform) write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgelsd", idgelsd, cput -&
                  & lcpu,&
                  & "solved Fourier", dlambda(1, 0)
            case (:-1)
              write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgelsd", idgelsd, cput - lcpu, "input error"
            case (1:)
              write(ounit, 1030)cput - cpus, myid, lvol, innout, id, "idgelsd", idgelsd, cput - lcpu, "QR failed"
            case default
              call fatal(.true., __FILE__, "illegal ifail returned by f04arf ")

            end select

            call fatal(idgelsd /= 0, __FILE__, "failed to construct straight-fieldline angle using dgelsd ")

            dmatrix(1:NN, 1:NN, 0) = omatrix(1:NN, 1:NN)! original "unperturbed" matrix

          case default

            call fatal(.true., __FILE__, "illegal Lsvdiota ")

          end select

          1030 format("compute_rotational_transform : ", f10.2, " ; myid=", i3, " ; lvol=", i3, " ; innout=", i2, " ; jderiv=", &
              i2, " ; ", a7, "=", i2, " ; time=", f10.4, " ; ", a17, :, " [d]iota=", es17.09, " ")
        end do

      end if

      if (Lsparse == 0 .or. Lsparse == 3) then
        deallocate(dmatrix)
        deallocate(omatrix)
        deallocate(FAA)
      end if
    end do
  end subroutine compute_rotational_transform
end module magnetic_field_mod
