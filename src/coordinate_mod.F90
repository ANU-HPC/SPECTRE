module coordinate_mod
  use precisions_mod, only: rkind
  use io_mod, only: ounit
  use input_list_mod, only: Wrzaxis, Igeometry, Ntor, Lcheck, Wmacros, Lreflect, Ntoraxis, Lrzaxis, enforce_stell_sym
  use allglobal_mod, only: myid, im, cpus, in, ajk, Rij, sg, ijreal, ijimag, jireal, jiimag, &
      jkreal, jkimag, kjreal, kjimag, efmn, ofmn, cfmn, sfmn, evmn, odmn, comn, simn, cosi, sini, &
      Lcoordinatesingularity, dRodR, dRodZ, dZodR, dZodZ, dRadR, dRadZ, dZadR, dZadZ, iRbc, iZbs, iRbs, &
      iZbc, dBdX, mn
  use memory_allocation_mod, only: sallocate
  use geometry_mod, only: compute_coordinates
  use numerical_mod, only: tfft, invfft

  implicit none(type, external)

  private
  public :: rzaxis, get_position_vetor_x, calc_axis_deriv_grid

  external :: DGETRF, DGETRS

contains

  subroutine rzaxis(Mvol, mn,  Nt, Nz, inRbc, inZbs, inRbs, inZbc, ivol, LcomputeDerivatives)

    logical, intent(in) :: LComputeDerivatives ! indicates whether derivatives are to be calculated

    integer, intent(in) :: Mvol, mn, ivol, Nt, Nz
    real(rkind) :: inRbc(1:mn, 0:Mvol), inZbs(1:mn, 0:Mvol), inRbs(1:mn, 0:Mvol), inZbc(1:mn, 0:Mvol)
    real(rkind) :: jRbc(1:mn, 0:Mvol), jZbs(1:mn, 0:Mvol), jRbs(1:mn, 0:Mvol), jZbc(1:mn, 0:Mvol)
    real(rkind) :: tmpRbc(1:mn, 0:Mvol), tmpZbs(1:mn, 0:Mvol), tmpRbs(1:mn, 0:Mvol), tmpZbc(1:mn, 0:Mvol)! use as temp
    ! matrices to store iRbc etc

    real(rkind) :: cput

    real(rkind) :: jacbase(1:(Nt*Nz)), jacbasec(1:mn), jacbases(1:mn)! the 2D Jacobian and its Fourier
    real(rkind) :: junkc(1:mn), junks(1:mn)! these are junk matrices used for fft

    integer :: jvol, ii, ifail, jj, id, issym, irz, imn
    integer :: idJc, idJs, idRc, idRs, idZc, idZs

    integer :: Lcurvature

    integer :: Njac, idgetrf, idgetrs ! internal variables used in Jacobian method
    real(rkind), allocatable :: jacrhs(:), djacrhs(:), jacmat(:, :), djacmat(:, :), solution(:), LU(:, :)! internal
    ! matrices used in Jacobian method
    integer, allocatable :: ipiv(:)! internal matrices used in  Jacobian method

    integer :: Ntz

    Ntz = Nt * Nz

    jvol = 0 ! this identifies the "surface" in which the poloidal averaged harmonics will be placed

    Ntoraxis = min(Ntor, Ntoraxis)

    select case (Igeometry)

    case (1:2)

      inRbc(1:mn, jvol) = 0.0_rkind
      inRbs(1:mn, jvol) = 0.0_rkind

      if (Igeometry == 1 .and. Lreflect == 1) then ! reflect upper and lower bound in slab, each take half the amplitude
        inRbc(2:mn, 0) = -inRbc(2:mn, Mvol)
        if (.not.enforce_stell_sym) then
          inRbs(2:mn, 0) = -inRbs(2:mn, Mvol)
        end if
      end if

    case (3)

      if (Lrzaxis == 1) then ! use centroid method

        call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * inRbs(1:mn, ivol), -im(1:mn) * inRbc(1:mn, ivol), im(1:mn) *&
            & inZbs(1:mn, ivol), -im(1:mn) * inZbc(1:mn, ivol), jkreal(1:Ntz), jkimag(1:Ntz))! R_\t, Z_\t

        ijreal(1:Ntz) = sqrt(jkreal(1:Ntz) ** 2 + jkimag(1:Ntz) ** 2)! dl
        ijimag(1:Ntz) = 0.0_rkind

        jireal(1:Ntz) = ijreal(1:Ntz)! dl

        ifail = 0
        call tfft(ijreal(1:Ntz), ijimag(1:Ntz), mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn),&
            & sfmn(1:mn), ifail)! Fourier harmonics of differential poloidal length

        efmn(1:mn) = efmn(1:mn) * ajk(1:mn)! poloidal integration of length: only take m=0 harmonics
        ofmn(1:mn) = ofmn(1:mn) * ajk(1:mn)
        cfmn(1:mn) = 0.0_rkind
        sfmn(1:mn) = 0.0_rkind

        call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), & ! map length = "integrated
        ! dl" back to real space
            ijreal(1:Ntz), ijimag(1:Ntz))

        jiimag(1:Ntz) = ijreal(1:Ntz)!  L

        call invfft(mn, im(1:mn), in(1:mn), inRbc(1:mn, ivol), inRbs(1:mn, ivol), inZbc(1:mn, ivol), inZbs(1:mn, ivol),&
            & kjreal(1:Ntz), kjimag(1:Ntz))! R, Z

        ijreal(1:Ntz) = kjreal(1:Ntz) * jireal(1:Ntz)! R dl
        ijimag(1:Ntz) = kjimag(1:Ntz) * jireal(1:Ntz)! Z dl

        ifail = 0
        call tfft(ijreal(1:Ntz), ijimag(1:Ntz), mn, im(1:mn), in(1:mn), evmn(1:mn), odmn(1:mn), comn(1:mn),&
            & simn(1:mn), ifail)! Fourier harmonics of weighted R & Z

        evmn(1:mn) = evmn(1:mn) * ajk(1:mn)! poloidal integration of R dl
        odmn(1:mn) = odmn(1:mn) * ajk(1:mn)
        comn(1:mn) = comn(1:mn) * ajk(1:mn)! poloidal integration of Z dl
        simn(1:mn) = simn(1:mn) * ajk(1:mn)

        call invfft(mn, im(1:mn), in(1:mn), evmn(1:mn), odmn(1:mn), comn(1:mn), simn(1:mn), ijreal(1:Ntz),&
            & ijimag(1:Ntz))

        ijreal(1:Ntz) = ijreal(1:Ntz) / jiimag(1:Ntz)! Ro
        ijimag(1:Ntz) = ijimag(1:Ntz) / jiimag(1:Ntz)! Zo

        kjreal(1:Ntz) = kjreal(1:Ntz) - ijreal(1:Ntz)! \Delta R = R_1 - R_0
        kjimag(1:Ntz) = kjimag(1:Ntz) - ijimag(1:Ntz)! \Delta R = Z_1 - Z_0

        ifail = 0
        call tfft(ijreal(1:Ntz), ijimag(1:Ntz), mn, im(1:mn), in(1:mn), inRbc(1:mn, jvol), inRbs(1:mn, jvol),&
            & inZbc(1:mn, jvol), inZbs(1:mn, jvol), ifail)

        if (LComputeDerivatives) then
          ! compute derivatives of axis

          do ii = 1, mn

            dRodR(1:Ntz, 0, ii) = cosi(1:Ntz, ii) * jireal(1:Ntz) - kjreal(1:Ntz) * jkreal(1:Ntz) * im(ii) * sini(1:Ntz,&
                & ii) / jireal(1:Ntz)! dRodRjc
            dRodR(1:Ntz, 1, ii) = sini(1:Ntz, ii) * jireal(1:Ntz) + kjreal(1:Ntz) * jkreal(1:Ntz) * im(ii) * cosi(1:Ntz,&
                & ii) / jireal(1:Ntz)! dRodRjs

            ifail = 0
            call tfft(dRodR(1:Ntz, 0, ii), dRodR(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dRadR(1:mn, 0, 0, ii),&
                & dRadR(1:mn, 1, 0, ii), dRadR(1:mn, 0, 1, ii), dRadR(1:mn, 1, 1, ii), ifail)

            dRadR(1:mn, 0, 0, ii) = dRadR(1:mn, 0, 0, ii) * ajk(1:mn)! poloidal integration
            dRadR(1:mn, 1, 0, ii) = dRadR(1:mn, 1, 0, ii) * ajk(1:mn)
            dRadR(1:mn, 0, 1, ii) = dRadR(1:mn, 0, 1, ii) * ajk(1:mn)
            dRadR(1:mn, 1, 1, ii) = dRadR(1:mn, 1, 1, ii) * ajk(1:mn)

            call invfft(mn, im(1:mn), in(1:mn), dRadR(1:mn, 0, 0, ii), dRadR(1:mn, 1, 0, ii), dRadR(1:mn, 0, 1, ii),&
                & dRadR(1:mn, 1, 1, ii), dRodR(1:Ntz, 0, ii), dRodR(1:Ntz, 1, ii))! R, Z

            dRodR(1:Ntz, 0, ii) = dRodR(1:Ntz, 0, ii) / jiimag(1:Ntz)! divide by length
            dRodR(1:Ntz, 1, ii) = dRodR(1:Ntz, 1, ii) / jiimag(1:Ntz)

            dRodZ(1:Ntz, 0, ii) = -kjreal(1:Ntz) * jkimag(1:Ntz) * im(ii) * sini(1:Ntz, ii) / jireal(1:Ntz)! dRodZjc
            dRodZ(1:Ntz, 1, ii) = +kjreal(1:Ntz) * jkimag(1:Ntz) * im(ii) * cosi(1:Ntz, ii) / jireal(1:Ntz)! dRodZjs

            ifail = 0
            call tfft(dRodZ(1:Ntz, 0, ii), dRodZ(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dRadZ(1:mn, 0, 0, ii),&
                & dRadZ(1:mn, 1, 0, ii), dRadZ(1:mn, 0, 1, ii), dRadZ(1:mn, 1, 1, ii), ifail)

            dRadZ(1:mn, 0, 0, ii) = dRadZ(1:mn, 0, 0, ii) * ajk(1:mn)! poloidal integration
            dRadZ(1:mn, 1, 0, ii) = dRadZ(1:mn, 1, 0, ii) * ajk(1:mn)
            dRadZ(1:mn, 0, 1, ii) = dRadZ(1:mn, 0, 1, ii) * ajk(1:mn)
            dRadZ(1:mn, 1, 1, ii) = dRadZ(1:mn, 1, 1, ii) * ajk(1:mn)

            call invfft(mn, im(1:mn), in(1:mn), dRadZ(1:mn, 0, 0, ii), dRadZ(1:mn, 1, 0, ii), dRadZ(1:mn, 0, 1, ii),&
                & dRadZ(1:mn, 1, 1, ii), dRodZ(1:Ntz, 0, ii), dRodZ(1:Ntz, 1, ii))! R, Z

            dRodZ(1:Ntz, 0, ii) = dRodZ(1:Ntz, 0, ii) / jiimag(1:Ntz)! divide by length
            dRodZ(1:Ntz, 1, ii) = dRodZ(1:Ntz, 1, ii) / jiimag(1:Ntz)

            dZodR(1:Ntz, 0, ii) = -kjimag(1:Ntz) * jkreal(1:Ntz) * im(ii) * sini(1:Ntz, ii) / jireal(1:Ntz)! dZodRjc
            dZodR(1:Ntz, 1, ii) = +kjimag(1:Ntz) * jkreal(1:Ntz) * im(ii) * cosi(1:Ntz, ii) / jireal(1:Ntz)! dZodRjs

            ifail = 0
            call tfft(dZodR(1:Ntz, 0, ii), dZodR(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dZadR(1:mn, 0, 0, ii),&
                & dZadR(1:mn, 1, 0, ii), dZadR(1:mn, 0, 1, ii), dZadR(1:mn, 1, 1, ii), ifail)

            dZadR(1:mn, 0, 0, ii) = dZadR(1:mn, 0, 0, ii) * ajk(1:mn)! poloidal integration
            dZadR(1:mn, 1, 0, ii) = dZadR(1:mn, 1, 0, ii) * ajk(1:mn)
            dZadR(1:mn, 0, 1, ii) = dZadR(1:mn, 0, 1, ii) * ajk(1:mn)
            dZadR(1:mn, 1, 1, ii) = dZadR(1:mn, 1, 1, ii) * ajk(1:mn)

            call invfft(mn, im(1:mn), in(1:mn), dZadR(1:mn, 0, 0, ii), dZadR(1:mn, 1, 0, ii), dZadR(1:mn, 0, 1, ii),&
                & dZadR(1:mn, 1, 1, ii), dZodR(1:Ntz, 0, ii), dZodR(1:Ntz, 1, ii))! R, Z

            dZodR(1:Ntz, 0, ii) = dZodR(1:Ntz, 0, ii) / jiimag(1:Ntz)! divide by length
            dZodR(1:Ntz, 1, ii) = dZodR(1:Ntz, 1, ii) / jiimag(1:Ntz)

            dZodZ(1:Ntz, 0, ii) = cosi(1:Ntz, ii) * jireal(1:Ntz) - kjimag(1:Ntz) * jkimag(1:Ntz) * im(ii) * sini(1:Ntz,&
                & ii) / jireal(1:Ntz)! dZodZjc
            dZodZ(1:Ntz, 1, ii) = sini(1:Ntz, ii) * jireal(1:Ntz) + kjimag(1:Ntz) * jkimag(1:Ntz) * im(ii) * cosi(1:Ntz,&
                & ii) / jireal(1:Ntz)! dZodZjs

            ifail = 0
            call tfft(dZodZ(1:Ntz, 0, ii), dZodZ(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dZadZ(1:mn, 0, 0, ii),&
                & dZadZ(1:mn, 1, 0, ii), dZadZ(1:mn, 0, 1, ii), dZadZ(1:mn, 1, 1, ii), ifail)

            dZadZ(1:mn, 0, 0, ii) = dZadZ(1:mn, 0, 0, ii) * ajk(1:mn)! poloidal integration
            dZadZ(1:mn, 1, 0, ii) = dZadZ(1:mn, 1, 0, ii) * ajk(1:mn)
            dZadZ(1:mn, 0, 1, ii) = dZadZ(1:mn, 0, 1, ii) * ajk(1:mn)
            dZadZ(1:mn, 1, 1, ii) = dZadZ(1:mn, 1, 1, ii) * ajk(1:mn)

            call invfft(mn, im(1:mn), in(1:mn), dZadZ(1:mn, 0, 0, ii), dZadZ(1:mn, 1, 0, ii), dZadZ(1:mn, 0, 1, ii),&
                & dZadZ(1:mn, 1, 1, ii), dZodZ(1:Ntz, 0, ii), dZodZ(1:Ntz, 1, ii))! R, Z

            dZodZ(1:Ntz, 0, ii) = dZodZ(1:Ntz, 0, ii) / jiimag(1:Ntz)! divide by length
            dZodZ(1:Ntz, 1, ii) = dZodZ(1:Ntz, 1, ii) / jiimag(1:Ntz)

            imn = ii

            call tfft(dRodR(1:Ntz, 0, ii), dRodR(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dRadR(1:mn, 0, 0, ii),&
                & dRadR(1:mn, 1, 0, ii), dRadR(1:mn, 0, 1, ii), dRadR(1:mn, 1, 1, ii), ifail)
            call tfft(dRodZ(1:Ntz, 0, ii), dRodZ(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dRadZ(1:mn, 0, 0, ii),&
                & dRadZ(1:mn, 1, 0, ii), dRadZ(1:mn, 0, 1, ii), dRadZ(1:mn, 1, 1, ii), ifail)
            call tfft(dZodR(1:Ntz, 0, ii), dZodR(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dZadR(1:mn, 0, 0, ii),&
                & dZadR(1:mn, 1, 0, ii), dZadR(1:mn, 0, 1, ii), dZadR(1:mn, 1, 1, ii), ifail)
            call tfft(dZodZ(1:Ntz, 0, ii), dZodZ(1:Ntz, 1, ii), mn, im(1:mn), in(1:mn), dZadZ(1:mn, 0, 0, ii),&
                & dZadZ(1:mn, 1, 0, ii), dZadZ(1:mn, 0, 1, ii), dZadZ(1:mn, 1, 1, ii), ifail)

            ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadR(1:mn, 1, 0, imn), +in(1:mn) * dRadR(1:mn, 0, 0, imn), &
            !     &-in(1:mn) * dRadR(1:mn, 1, 1, imn), +in(1:mn) * dRadR(1:mn, 0, 1, imn), dRodR(1:Ntz, 2, imn),&
            !     & dRodR(1:Ntz, 3, imn))
            ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadZ(1:mn, 1, 0, imn), +in(1:mn) * dRadZ(1:mn, 0, 0, imn), &
            !     &-in(1:mn) * dRadZ(1:mn, 1, 1, imn), +in(1:mn) * dRadZ(1:mn, 0, 1, imn), dRodZ(1:Ntz, 2, imn),&
            !     & dRodZ(1:Ntz, 3, imn))
            ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadR(1:mn, 1, 0, imn), +in(1:mn) * dZadR(1:mn, 0, 0, imn), &
            !     &-in(1:mn) * dZadR(1:mn, 1, 1, imn), +in(1:mn) * dZadR(1:mn, 0, 1, imn), dZodR(1:Ntz, 2, imn),&
            !     & dZodR(1:Ntz, 3, imn))
            ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadZ(1:mn, 1, 0, imn), +in(1:mn) * dZadZ(1:mn, 0, 0, imn), &
            !     &-in(1:mn) * dZadZ(1:mn, 1, 1, imn), +in(1:mn) * dZadZ(1:mn, 0, 1, imn), dZodZ(1:Ntz, 2, imn),&
            !     & dZodZ(1:Ntz, 3, imn))
          end do
        end if !if (LComputeDerivatives) then
      else if (Lrzaxis == 2) then ! use Jacobian m=1 harmonic elimination method

        ! We need to back up a few things before we proceed.
        ! The complication is, iRbc and inRbc could refer to the same thing, due to the way rzaxis is used.
        ! Once iRbc is changed, inRbc could also change, so we should seperate them as two variables.
        tmpRbc = iRbc
        tmpZbs = iZbs
        tmpRbs = iRbs
        tmpZbc = iZbc

        jRbc = inRbc
        jZbs = inZbs
        jRbs = inRbs
        jZbc = inZbc
        ! Now, inRbc should never be used until the end of subroutine in which they get filled back

        ! determine the number of equations and initialize the matrix, temp variables and rhs
        if (enforce_stell_sym) then
          Njac = 2 * Ntoraxis + 1
        else
          Njac = 2 * (2 * Ntoraxis + 1)
        end if

        call sallocate(jacrhs, 1, Njac, 0.0_rkind)
        call sallocate(jacmat, 1, Njac, 1, Njac, 0.0_rkind)
        call sallocate(LU, 1, Njac, 1, Njac, 0.0_rkind)
        call sallocate(solution, 1, Njac, 0.0_rkind)
        call sallocate(ipiv, 1, Njac, 0)

        ! replace iRbc to use subroutine compute_coordinates
        iRbc(1:mn, 1) = jRbc(1:mn, ivol)
        iZbs(1:mn, 1) = jZbs(1:mn, ivol)
        iRbs(1:mn, 1) = jRbs(1:mn, ivol)
        iZbc(1:mn, 1) = jZbc(1:mn, ivol)

        iRbc(1:mn, 0) = 0.0_rkind
        iZbs(1:mn, 0) = 0.0_rkind
        iRbs(1:mn, 0) = 0.0_rkind
        iZbc(1:mn, 0) = 0.0_rkind

        iRbc(1:Ntor + 1, 0) = jRbc(1:Ntor + 1, ivol)
        iZbs(1:Ntor + 1, 0) = jZbs(1:Ntor + 1, ivol)
        iRbs(1:Ntor + 1, 0) = jRbs(1:Ntor + 1, ivol)
        iZbc(1:Ntor + 1, 0) = jZbc(1:Ntor + 1, ivol)

        Lcoordinatesingularity = .true.
        Lcurvature = 1
        dBdX%innout = 1

        ! the indices in the matrix
        idJc = Ntoraxis + 1 ! rhs index for J cos n=0 term
        idJs = Ntoraxis + 1 + 2 * Ntoraxis + 1 ! rhs index for J sin n=0 term
        idRc = 1
        idZs = Ntoraxis + 1
        idRs = 2 * Ntoraxis + 1
        idZc = 3 * Ntoraxis + 2

        call compute_coordinates(1, 1.0_rkind, Lcurvature, Nt, Nz, mn)

        jacbase = sg(1:Ntz, 0) / Rij(1:Ntz, 0, 0)! extract the baseline 2D jacobian, note the definition here does not
        ! have the R factor

        call tfft(jacbase, Rij, mn, im(1:mn), in(1:mn), jacbasec(1:mn), jacbases(1:mn), junkc(1:mn), junks(1:mn),&
            & ifail)

        ! fill in the right hand side with m=1 terms of Jacobian
        if (enforce_stell_sym) then
          jacrhs = -jacbasec(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
        else
          jacrhs(1:2 * Ntoraxis + 1) = -jacbasec(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
          jacrhs(2 * Ntoraxis + 2:Njac) = -jacbases(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
        end if !if (enforce_stell_sym)

        if (enforce_stell_sym) then

          do ii = -Ntoraxis, Ntoraxis
            do jj = 1, Ntoraxis

              if (ii - jj >= -Ntor) then
                id = 2 * (Ntor + 1) + ii - jj
                ! the DRcn' term
                jacmat(ii + Ntoraxis + 1, jj + 1) = jacmat(ii + Ntoraxis + 1, jj + 1) - jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) = jacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) + jRbc(id,&
                    & ivol)
              end if ! if (ii-jj .ge. -Ntor)

              if (ii + jj <= Ntor) then
                id = 2 * (Ntor + 1) + ii + jj
                ! the DRcn' term
                jacmat(ii + Ntoraxis + 1, jj + 1) = jacmat(ii + Ntoraxis + 1, jj + 1) - jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) = jacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) - jRbc(id,&
                    & ivol)
              end if ! if (ii+jj .le. Ntor)

            end do ! jj

            ! the DR0 term
            id = 2 * (Ntor + 1) + ii
            jacmat(ii + Ntoraxis + 1, 1) = -2.0_rkind * jZbs(id, ivol)

          end do ! ii

        else ! for .not.enforce_stell_sym

          do ii = -Ntoraxis, Ntoraxis
            do jj = 1, Ntoraxis

              if (ii - jj >= -Ntor) then
                id = 2 * (Ntor + 1) + ii - jj
                ! for J cos terms
                ! the DRcn' term
                jacmat(ii + idJc, jj + idRc) = jacmat(ii + idJc, jj + idRc) - jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJc, jj + idZs) = jacmat(ii + idJc, jj + idZs) + jRbc(id, ivol)
                ! the DRsn' term
                jacmat(ii + idJc, jj + idRs) = jacmat(ii + idJc, jj + idRs) - jZbc(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJc, jj + idZc) = jacmat(ii + idJc, jj + idZc) + jRbs(id, ivol)

                ! for J sin terms
                ! the DRcn' term
                jacmat(ii + idJs, jj + idRc) = jacmat(ii + idJs, jj + idRc) + jZbc(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJs, jj + idZs) = jacmat(ii + idJs, jj + idZs) + jRbs(id, ivol)
                ! the DRsn' term
                jacmat(ii + idJs, jj + idRs) = jacmat(ii + idJs, jj + idRs) - jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJs, jj + idZc) = jacmat(ii + idJs, jj + idZc) - jRbc(id, ivol)

              end if ! if (ii-jj .ge. -Ntor)

              if (ii + jj <= Ntor) then
                id = 2 * (Ntor + 1) + ii + jj

                ! for J cos terms
                ! the DRcn' term
                jacmat(ii + idJc, jj + idRc) = jacmat(ii + idJc, jj + idRc) - jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJc, jj + idZs) = jacmat(ii + idJc, jj + idZs) - jRbc(id, ivol)
                ! the DRsn' term
                jacmat(ii + idJc, jj + idRs) = jacmat(ii + idJc, jj + idRs) + jZbc(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJc, jj + idZc) = jacmat(ii + idJc, jj + idZc) + jRbs(id, ivol)

                ! for J sin terms
                ! the DRcn' term
                jacmat(ii + idJs, jj + idRc) = jacmat(ii + idJs, jj + idRc) + jZbc(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJs, jj + idZs) = jacmat(ii + idJs, jj + idZs) - jRbs(id, ivol)
                ! the DRsn' term
                jacmat(ii + idJs, jj + idRs) = jacmat(ii + idJs, jj + idRs) + jZbs(id, ivol)
                ! the DZsn' term
                jacmat(ii + idJs, jj + idZc) = jacmat(ii + idJs, jj + idZc) - jRbc(id, ivol)
              end if ! if (ii+jj .le. Ntor)

            end do ! jj

            ! the DR0 term
            id = 2 * (Ntor + 1) + ii
            ! for J cos terms
            jacmat(ii + idJc, idRc) = -2.0_rkind * jZbs(id, ivol)
            jacmat(ii + idJc, idZc) = +2.0_rkind * jRbs(id, ivol)
            ! for J sin terms
            jacmat(ii + idJs, idRc) = +2.0_rkind * jZbc(id, ivol)
            jacmat(ii + idJs, idZc) = -2.0_rkind * jRbc(id, ivol)

          end do ! ii

        end if ! if (enforce_stell_sym)

        jacmat = jacmat * 1.0_rkind / 2.0_rkind ! because we are using (1+s)/2 instead of s

        LU = jacmat
        call DGETRF(Njac, Njac, LU, Njac, ipiv, idgetrf)! LU factorization
        solution = jacrhs
        call DGETRS('N', Njac, 1, LU, Njac, ipiv, solution, Njac, idgetrs)! sovle linear equation

        if (idgetrf < 0 .or. idgetrs < 0) then
          write(ounit, 1010)cput - cpus, myid, ivol, idgetrf, idgetrs, "input error"
        elseif (idgetrf > 0) then
          write(ounit, 1010)cput - cpus, myid, ivol, idgetrf, idgetrs, "singular"
        end if

        1010 format("rzaxis : ", f10.2, " : myid=", i3, " ; ivol=", i3, " idgetrf idgetrs=", i3, ' ', i3, "  ", a34)

        if (LComputeDerivatives) then
          ! copy the data to jRbc etc

          jRbc(1:Ntor + 1, jvol) = 0.0_rkind
          jZbs(2:Ntor + 1, jvol) = 0.0_rkind
          jRbc(1:Ntoraxis + 1, jvol) = jRbc(1:Ntoraxis + 1, ivol) - solution(1:Ntoraxis + 1)
          jZbs(2:Ntoraxis + 1, jvol) = jZbs(2:Ntoraxis + 1, ivol) - solution(Ntoraxis + 2:2 * Ntoraxis + 1)
          if (.not.enforce_stell_sym) then
            jRbs(2:Ntor + 1, jvol) = 0.0_rkind
            jZbc(1:Ntor + 1, jvol) = 0.0_rkind
            jRbs(2:Ntoraxis + 1, jvol) = jRbs(2:Ntoraxis + 1, ivol) - solution(idRs + 1:idRs + Ntoraxis)
            jZbc(1:Ntoraxis + 1, jvol) = jZbc(1:Ntoraxis + 1, ivol) - solution(idZc:idZc + Ntoraxis)
          end if ! .not.enforce_stell_sym

          ! compute the derivative w.r.t. Rjc, Rjs, Zjc, Zjs using matrix perturbation theory
          ! clean up the result from last time
          dRodR = 0.0_rkind
          dRodZ = 0.0_rkind
          dZodR = 0.0_rkind
          dZodZ = 0.0_rkind
          dRadR = 0.0_rkind
          dRadZ = 0.0_rkind
          dZadR = 0.0_rkind
          dZadZ = 0.0_rkind

          ! allocate the temp matrices
          call sallocate(djacrhs, 1, Njac, 0.0_rkind)
          call sallocate(djacmat, 1, Njac, 1, Njac, 0.0_rkind)

          dBdX%L = .true. ! will need derivatives

          do imn = 1, mn ! loop over deformations in Fourier harmonics: inside do vvol
            dBdX%ii = imn ! controls construction of derivatives in subroutines called below
            do irz = 0, 1 ! loop over deformations in R and Z: inside do imn
              dBdX%irz = irz ! controls construction of derivatives
              do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                if (imn == 1 .and. irz /= issym) cycle ! no m=n=0 sin terms
                dBdX%issym = issym ! controls construction of derivatives

                ! clean up for every new loop
                djacmat = 0.0_rkind
                djacrhs = 0.0_rkind

                Lcoordinatesingularity = .true.
                Lcurvature = 5 ! specially designed to drive subroutine "compute_coordinates" to compute 2D jacobian derivative w.r.t.
                ! interface

                if (im(imn) == 0) then ! the jacobian on the RHS does not depend on m=0 terms
                  jacbase = 0.0_rkind
                else
                  call compute_coordinates(1, 1.0_rkind, Lcurvature, Nt, Nz, mn)! the derivative of Jabobian w.r.t. geometry is computed
                  ! by compute_coordinates
                  jacbase = sg(1:Ntz, 1)
                end if

                call tfft(jacbase, Rij, mn, im(1:mn), in(1:mn), jacbasec(1:mn), jacbases(1:mn), junkc(1:mn),&
                    & junks(1:mn), ifail)

                if (enforce_stell_sym) then
                  djacrhs = -jacbasec(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
                else
                  djacrhs(1:2 * Ntoraxis + 1) = -jacbasec(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
                  djacrhs(2 * Ntoraxis + 2:Njac) = -jacbases(2 * (Ntor + 1) - Ntoraxis:2 * (Ntor + 1) + Ntoraxis)
                end if !if (enforce_stell_sym)

                if (im(imn) == 1) then ! djacmat for m=1 terms

                  if (enforce_stell_sym) then

                    do ii = -Ntoraxis, Ntoraxis
                      do jj = 1, Ntoraxis

                        if (ii - jj >= -Ntor) then
                          id = 2 * (Ntor + 1) + ii - jj
                          ! the DRcn' term
                          if (id == imn .and. irz == 1) djacmat(ii + Ntoraxis + 1, jj + 1) = djacmat(ii + Ntoraxis + 1,&
                              & jj + 1) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0) djacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) = djacmat(ii +&
                              & Ntoraxis + 1, Ntoraxis + 1 + jj) + 1.0_rkind
                        end if ! if (ii-jj .ge. -Ntor)

                        if (ii + jj <= Ntor) then
                          id = 2 * (Ntor + 1) + ii + jj
                          ! the DRcn' term
                          if (id == imn .and. irz == 1) djacmat(ii + Ntoraxis + 1, jj + 1) = djacmat(ii + Ntoraxis + 1,&
                              & jj + 1) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0) djacmat(ii + Ntoraxis + 1, Ntoraxis + 1 + jj) = djacmat(ii +&
                              & Ntoraxis + 1, Ntoraxis + 1 + jj) - 1.0_rkind
                        end if ! if (ii+jj .le. Ntor)

                      end do ! jj

                      ! the DR0 term
                      id = 2 * (Ntor + 1) + ii
                      if (id == imn .and. irz == 1) djacmat(ii + Ntoraxis + 1, 1) = -2.0_rkind

                    end do ! ii
                  else ! if (enforce_stell_sym)
                    do ii = -Ntoraxis, Ntoraxis
                      do jj = 1, Ntoraxis

                        if (ii - jj >= -Ntor) then
                          id = 2 * (Ntor + 1) + ii - jj
                          ! for J cos terms
                          ! the DRcn' term
                          if (id == imn .and. irz == 1 .and. issym == 0) djacmat(ii + idJc, jj + idRc) = djacmat(ii +&
                              & idJc, jj + idRc) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 0) djacmat(ii + idJc, jj + idZs) = djacmat(ii +&
                              & idJc, jj + idZs) + 1.0_rkind
                          ! the DRsn' term
                          if (id == imn .and. irz == 1 .and. issym == 1) djacmat(ii + idJc, jj + idRs) = djacmat(ii +&
                              & idJc, jj + idRs) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 1) djacmat(ii + idJc, jj + idZc) = djacmat(ii +&
                              & idJc, jj + idZc) + 1.0_rkind

                          ! for J sin terms
                          ! the DRcn' term
                          if (id == imn .and. irz == 1 .and. issym == 1) djacmat(ii + idJs, jj + idRc) = djacmat(ii +&
                              & idJs, jj + idRc) + 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 1) djacmat(ii + idJs, jj + idZs) = djacmat(ii +&
                              & idJs, jj + idZs) + 1.0_rkind
                          ! the DRsn' term
                          if (id == imn .and. irz == 1 .and. issym == 0) djacmat(ii + idJs, jj + idRs) = djacmat(ii +&
                              & idJs, jj + idRs) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 0) djacmat(ii + idJs, jj + idZc) = djacmat(ii +&
                              & idJs, jj + idZc) - 1.0_rkind

                        end if ! if (ii-jj .ge. -Ntor)

                        if (ii + jj <= Ntor) then
                          id = 2 * (Ntor + 1) + ii + jj
                          ! for J cos terms
                          ! the DRcn' term
                          if (id == imn .and. irz == 1 .and. issym == 0) djacmat(ii + idJc, jj + idRc) = djacmat(ii +&
                              & idJc, jj + idRc) - 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 0) djacmat(ii + idJc, jj + idZs) = djacmat(ii +&
                              & idJc, jj + idZs) - 1.0_rkind
                          ! the DRsn' term
                          if (id == imn .and. irz == 1 .and. issym == 1) djacmat(ii + idJc, jj + idRs) = djacmat(ii +&
                              & idJc, jj + idRs) + 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 1) djacmat(ii + idJc, jj + idZc) = djacmat(ii +&
                              & idJc, jj + idZc) + 1.0_rkind

                          ! for J sin terms
                          ! the DRcn' term
                          if (id == imn .and. irz == 1 .and. issym == 1) djacmat(ii + idJs, jj + idRc) = djacmat(ii +&
                              & idJs, jj + idRc) + 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 1) djacmat(ii + idJs, jj + idZs) = djacmat(ii +&
                              & idJs, jj + idZs) - 1.0_rkind
                          ! the DRsn' term
                          if (id == imn .and. irz == 1 .and. issym == 0) djacmat(ii + idJs, jj + idRs) = djacmat(ii +&
                              & idJs, jj + idRs) + 1.0_rkind
                          ! the DZsn' term
                          if (id == imn .and. irz == 0 .and. issym == 0) djacmat(ii + idJs, jj + idZc) = djacmat(ii +&
                              & idJs, jj + idZc) - 1.0_rkind

                        end if ! if (ii+jj .le. Ntor)

                      end do ! jj

                      ! the DR0 term
                      id = 2 * (Ntor + 1) + ii
                      ! for J cos terms
                      if (id == imn .and. irz == 1 .and. issym == 0) djacmat(ii + idJc, idRc) = -2.0_rkind
                      if (id == imn .and. irz == 0 .and. issym == 1) djacmat(ii + idJc, idZc) = +2.0_rkind
                      ! for J sin terms
                      if (id == imn .and. irz == 1 .and. issym == 1) djacmat(ii + idJs, idRc) = +2.0_rkind
                      if (id == imn .and. irz == 0 .and. issym == 0) djacmat(ii + idJs, idZc) = -2.0_rkind

                    end do ! ii

                  end if ! if (enforce_stell_sym)

                  djacmat = djacmat * 1.0_rkind / 2.0_rkind ! because we are using (1+s)/2 instead of s

                  ! use matrix perturbation theory to compute the analytical derivatives
                  djacrhs = djacrhs - matmul(djacmat, solution)
                end if ! im(imn).eq.1

                call DGETRS('N', Njac, 1, LU, Njac, ipiv, djacrhs, Njac, idgetrs)! solve linear equation

                if (enforce_stell_sym) then
                  if (irz == 0) then
                    dRadR(1:Ntoraxis + 1, 0, 0, imn) = -djacrhs(1:Ntoraxis + 1)
                    dZadR(2:Ntoraxis + 1, 1, 0, imn) = -djacrhs(Ntoraxis + 2:Njac)
                    if (im(imn) == 0) then ! addtional 1.0_rkind
                      dRadR(imn, 0, 0, imn) = dRadR(imn, 0, 0, imn) + 1.0_rkind
                    end if
                  else
                    dRadZ(1:Ntoraxis + 1, 0, 1, imn) = -djacrhs(1:Ntoraxis + 1)
                    dZadZ(2:Ntoraxis + 1, 1, 1, imn) = -djacrhs(Ntoraxis + 2:Njac)
                    if (im(imn) == 0) then ! addtional 1.0_rkind
                      dZadZ(imn, 1, 1, imn) = dZadZ(imn, 1, 1, imn) + 1.0_rkind
                    end if
                  end if
                else
                  if (irz == 0) then
                    dRadR(1:Ntoraxis + 1, 0, issym, imn) = -djacrhs(idRc:idRc + Ntoraxis)
                    dZadR(2:Ntoraxis + 1, 1, issym, imn) = -djacrhs(idZs + 1:idZs + Ntoraxis)
                    dRadR(2:Ntoraxis + 1, 1, issym, imn) = -djacrhs(idRs + 1:idRs + Ntoraxis)
                    dZadR(1:Ntoraxis + 1, 0, issym, imn) = -djacrhs(idZc:idZc + Ntoraxis)
                    if (im(imn) == 0) then ! addtional 1.0_rkind
                      dRadR(imn, issym, issym, imn) = dRadR(imn, issym, issym, imn) + 1.0_rkind
                    end if

                  else
                    dRadZ(1:Ntoraxis + 1, 0, 1 - issym, imn) = -djacrhs(idRc:idRc + Ntoraxis)
                    dZadZ(2:Ntoraxis + 1, 1, 1 - issym, imn) = -djacrhs(idZs + 1:idZs + Ntoraxis)
                    dRadZ(2:Ntoraxis + 1, 1, 1 - issym, imn) = -djacrhs(idRs + 1:idRs + Ntoraxis)
                    dZadZ(1:Ntoraxis + 1, 0, 1 - issym, imn) = -djacrhs(idZc:idZc + Ntoraxis)
                    if (im(imn) == 0) then ! addtional 1.0_rkind
                      dZadZ(imn, 1 - issym, 1 - issym, imn) = dZadZ(imn, 1 - issym, 1 - issym, imn) + 1.0_rkind
                    end if
                  end if
                end if ! enforce_stell_sym

                ! call invfft(mn, im(1:mn), in(1:mn), dRadR(1:mn, 0, 0, imn), dRadR(1:mn, 1, 0, imn), dRadR(1:mn, 0, 1, &
                !     & imn), dRadR(1:mn, 1, 1, imn), dRodR(1:Ntz, 0, imn), dRodR(1:Ntz, 1, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), dRadZ(1:mn, 0, 0, imn), dRadZ(1:mn, 1, 0, imn), dRadZ(1:mn, 0, 1, &
                !     & imn), dRadZ(1:mn, 1, 1, imn), dRodZ(1:Ntz, 0, imn), dRodZ(1:Ntz, 1, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), dZadR(1:mn, 0, 0, imn), dZadR(1:mn, 1, 0, imn), dZadR(1:mn, 0, 1, &
                !     & imn), dZadR(1:mn, 1, 1, imn), dZodR(1:Ntz, 0, imn), dZodR(1:Ntz, 1, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), dZadZ(1:mn, 0, 0, imn), dZadZ(1:mn, 1, 0, imn), dZadZ(1:mn, 0, 1, &
                !     & imn), dZadZ(1:mn, 1, 1, imn), dZodZ(1:Ntz, 2, imn), dZodZ(1:Ntz, 1, imn))

                ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadR(1:mn, 1, 0, imn), +in(1:mn) * dRadR(1:mn, 0, 0, &
                !     & imn), -in(1:mn) * dRadR(1:mn, 1, 1, imn), +in(1:mn) * dRadR(1:mn, 0, 1, imn), dRodR(1:Ntz, &
                !     & 2, imn), dRodR(1:Ntz, 3, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadZ(1:mn, 1, 0, imn), +in(1:mn) * dRadZ(1:mn, 0, 0, &
                !     & imn), -in(1:mn) * dRadZ(1:mn, 1, 1, imn), +in(1:mn) * dRadZ(1:mn, 0, 1, imn), dRodZ(1:Ntz, &
                !     & 2, imn), dRodZ(1:Ntz, 3, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadR(1:mn, 1, 0, imn), +in(1:mn) * dZadR(1:mn, 0, 0, &
                !     & imn), -in(1:mn) * dZadR(1:mn, 1, 1, imn), +in(1:mn) * dZadR(1:mn, 0, 1, imn), dZodR(1:Ntz, &
                !     & 2, imn), dZodR(1:Ntz, 3, imn))
                ! call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadZ(1:mn, 1, 0, imn), +in(1:mn) * dZadZ(1:mn, 0, 0, &
                !     & imn), -in(1:mn) * dZadZ(1:mn, 1, 1, imn), +in(1:mn) * dZadZ(1:mn, 0, 1, imn), dZodZ(1:Ntz, &
                !     & 2, imn), dZodZ(1:Ntz, 3, imn))
              end do ! issym
            end do ! irz
          end do ! imn = 1, mn

          ! deallocate the matrices
          deallocate(djacrhs)
          deallocate(djacmat)

          dBdX%L = .false.

        end if ! if (LcomputeDerivatives)
        !******* End part used to check if the m=1 harmonics were successfully eliminated *******

        ! Clean up
        ! copy back the original data
        iRbc = tmpRbc
        iZbs = tmpZbs
        iRbs = tmpRbs
        iZbc = tmpZbc

        ! copy the data to output

        inRbc(:, jvol) = 0.0_rkind
        inZbs(:, jvol) = 0.0_rkind
        inRbs(:, jvol) = 0.0_rkind
        inZbc(:, jvol) = 0.0_rkind

        inRbc(1:Ntoraxis + 1, jvol) = inRbc(1:Ntoraxis + 1, ivol) - solution(idRc:idRc + Ntoraxis)
        inZbs(2:Ntoraxis + 1, jvol) = inZbs(2:Ntoraxis + 1, ivol) - solution(idZs + 1:idZs + Ntoraxis)
        if (enforce_stell_sym) then
          inRbs(1:Ntoraxis + 1, jvol) = 0.0_rkind
          inZbc(2:Ntoraxis + 1, jvol) = 0.0_rkind
        else
          inRbs(2:Ntoraxis + 1, jvol) = inRbs(2:Ntoraxis + 1, ivol) - solution(idRs + 1:idRs + Ntoraxis)
          inZbc(1:Ntoraxis + 1, jvol) = inZbc(1:Ntoraxis + 1, ivol) - solution(idZc:idZc + Ntoraxis)
        end if ! enforce_stell_sym

        ! Deallocate
        deallocate(jacrhs)
        deallocate(jacmat)
        deallocate(LU)
        deallocate(solution)
        deallocate(ipiv)
      end if

      if (LcomputeDerivatives) then
        call calc_axis_deriv_grid(Ntz)
      end if

    end select

  end subroutine rzaxis

  subroutine calc_axis_deriv_grid(ntz_loc)
    integer, intent(in) :: ntz_loc
    integer :: imn

    if (Igeometry /= 3) return

    do imn = 1, mn
      call invfft(mn, im(1:mn), in(1:mn), dRadR(1:mn, 0, 0, imn), dRadR(1:mn, 1, 0, imn), dRadR(1:mn, 0, 1, &
          & imn), dRadR(1:mn, 1, 1, imn), dRodR(1:ntz_loc, 0, imn), dRodR(1:ntz_loc, 1, imn))
      call invfft(mn, im(1:mn), in(1:mn), dRadZ(1:mn, 0, 0, imn), dRadZ(1:mn, 1, 0, imn), dRadZ(1:mn, 0, 1, &
          & imn), dRadZ(1:mn, 1, 1, imn), dRodZ(1:ntz_loc, 0, imn), dRodZ(1:ntz_loc, 1, imn))
      call invfft(mn, im(1:mn), in(1:mn), dZadR(1:mn, 0, 0, imn), dZadR(1:mn, 1, 0, imn), dZadR(1:mn, 0, 1, &
          & imn), dZadR(1:mn, 1, 1, imn), dZodR(1:ntz_loc, 0, imn), dZodR(1:ntz_loc, 1, imn))
      call invfft(mn, im(1:mn), in(1:mn), dZadZ(1:mn, 0, 0, imn), dZadZ(1:mn, 1, 0, imn), dZadZ(1:mn, 0, 1, &
          & imn), dZadZ(1:mn, 1, 1, imn), dZodZ(1:ntz_loc, 2, imn), dZodZ(1:ntz_loc, 1, imn))

      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadR(1:mn, 1, 0, imn), +in(1:mn) * dRadR(1:mn, 0, 0, &
          & imn), -in(1:mn) * dRadR(1:mn, 1, 1, imn), +in(1:mn) * dRadR(1:mn, 0, 1, imn), dRodR(1:ntz_loc, &
          & 2, imn), dRodR(1:ntz_loc, 3, imn))
      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dRadZ(1:mn, 1, 0, imn), +in(1:mn) * dRadZ(1:mn, 0, 0, &
          & imn), -in(1:mn) * dRadZ(1:mn, 1, 1, imn), +in(1:mn) * dRadZ(1:mn, 0, 1, imn), dRodZ(1:ntz_loc, &
          & 2, imn), dRodZ(1:ntz_loc, 3, imn))
      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadR(1:mn, 1, 0, imn), +in(1:mn) * dZadR(1:mn, 0, 0, &
          & imn), -in(1:mn) * dZadR(1:mn, 1, 1, imn), +in(1:mn) * dZadR(1:mn, 0, 1, imn), dZodR(1:ntz_loc, &
          & 2, imn), dZodR(1:ntz_loc, 3, imn))
      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * dZadZ(1:mn, 1, 0, imn), +in(1:mn) * dZadZ(1:mn, 0, 0, &
          & imn), -in(1:mn) * dZadZ(1:mn, 1, 1, imn), +in(1:mn) * dZadZ(1:mn, 0, 1, imn), dZodZ(1:ntz_loc, &
          & 2, imn), dZodZ(1:ntz_loc, 3, imn))
    end do

  end subroutine calc_axis_deriv_grid

  subroutine get_position_vetor_x(lvol, stz, RpZ)
    integer, intent(in) :: lvol

    real(rkind), intent(in) :: stz(1:3)
    real(rkind), intent(out) :: RpZ(1:3)

    integer :: ii, mi, ni
    real(rkind) :: Remn, Zomn, Romn, Zemn, RR, phi, ZZ, arg, carg, sarg, lss, alss, blss, sbar, fj

    RpZ(1:3) = 0.0_rkind
    RR = 0.0_rkind
    phi = stz(3)
    ZZ = 0.0_rkind ! initialize intent(out), summations

    lss = stz(1)
    alss = 1.0_rkind / 2.0_rkind - lss * 1.0_rkind / 2.0_rkind
    blss = 1.0_rkind / 2.0_rkind + lss * 1.0_rkind / 2.0_rkind ! shorthand

    do ii = 1, mn
      mi = im(ii)
      ni = in(ii)
      arg = mi * stz(2) - ni * phi
      carg = cos(arg)
      sarg = sin(arg)! Fourier sum

      Remn = 0.0_rkind
      Zomn = 0.0_rkind
      !  if( .not.enforce_stell_sym ) then
      Romn = 0.0_rkind
      Zemn = 0.0_rkind
      !  endif

      if (Lcoordinatesingularity) then

        sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind
        if (mi == 0) then
          if (Igeometry == 2) then
            fj = sbar
          else
            fj = sbar ** 2
          end if
        else
          fj = sbar ** im(ii)
        end if

        Remn = iRbc(ii, 0) + (iRbc(ii, 1) - iRbc(ii, 0)) * fj
        if (.not.enforce_stell_sym) then
          Romn = iRbs(ii, 0) + (iRbs(ii, 1) - iRbs(ii, 0)) * fj
        end if
        if (Igeometry == 3) then ! recall that for cylindrical geometry there is no need for Z
          Zomn = iZbs(ii, 0) + (iZbs(ii, 1) - iZbs(ii, 0)) * fj
          if (.not.enforce_stell_sym) then
            Zemn = iZbc(ii, 0) + (iZbc(ii, 1) - iZbc(ii, 0)) * fj
          end if
        end if

      else ! matches if( Lcoordinatesingularity)

        Remn = alss * iRbc(ii, lvol - 1) + blss * iRbc(ii, lvol)
        if (.not.enforce_stell_sym) then
          Romn = alss * iRbs(ii, lvol - 1) + blss * iRbs(ii, lvol)
        else
          Romn = 0.0_rkind
        end if
        if (Igeometry == 3) then
          Zomn = alss * iZbs(ii, lvol - 1) + blss * iZbs(ii, lvol)
          if (.not.enforce_stell_sym) then
            Zemn = alss * iZbc(ii, lvol - 1) + blss * iZbc(ii, lvol)
          else
            Zemn = 0.0_rkind
          end if
        else
          Zomn = 0.0_rkind
          Zemn = 0.0_rkind
        end if

      end if

      RR = RR + Remn * carg + Romn * sarg
      if (Igeometry == 3) then
        ZZ = ZZ + Zemn * carg + Zomn * sarg
      end if

    end do

    RpZ(1:3) = [RR, phi, ZZ] ! return coordinate functions
  end subroutine get_position_vetor_x
end module coordinate_mod
