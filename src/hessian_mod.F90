module hessian_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind
  use io_mod, only: stdout
  use input_list_mod, only: Wmacros, Whesian, Igeometry, Nvol, pflux, helicity, mu, Lfreebound, LHevalues, LHevectors,&
      & LHmatrix, dpp, dqq, enforce_stell_sym, Lcheck, Lfindzero, Lconstraint
  use allglobal_mod, only: myid, cpus, get_hidden, im, in, iRbc, iZbs, iRbs, iZbc, Energy, HdFFdRZ, dBBdmp, dmupfdx,&
      & force_ft_trunc_grad, force_ft_trunc_grad_alloc, psifactor, hessian2D, denergydrr, denergydzr, LocalConstraint, &
      Lhessian2Dallocated, Nt, Nz
  use sphdf5_mod, only: write_stability
  use memory_allocation_mod, only: sallocate
  use errors_mod, only: fatal
  use forces_mod, only: dforce
  use packing_mod, only: packxi

  implicit none(type, external)

  private
  public :: compute_hessian

  external :: dgetrf, DGEEV

contains

  subroutine compute_hessian(NGdof, position, Mvol, mn, LGdof)
    integer, intent(in) :: NGdof
    real(rkind), intent(inout) :: position(0:NGdof)
    integer, intent(in) :: Mvol
    integer, intent(in) :: mn
    integer, intent(in) :: LGdof

    real(rkind) :: cput
    logical :: LComputeDerivatives, LComputeAxis

    real(rkind) :: force(0:NGdof)

    real(rkind) :: xx(0:NGdof, -2:2), ff(0:NGdof, -2:2), df(1:NGdof)

    integer :: vvol, ii, irz, issym, isymdiff, lvol
    real(rkind) :: oldEnergy(-2:2), error, cpul

    real(rkind) :: ohessian(1:NGdof, 1:NGdof)

    real(rkind) :: determinant

    character :: pack

    character :: svol * 3

    integer :: tdof, tdoc, jvol, jj, jrz, jssym

    integer :: Lwork, LDA, Ldvi, Ldvr, if02ebf
    real(rkind) :: evalr(1:NGdof), evali(1:NGdof)
    real(rkind) :: evecr(1:NGdof, 1:NGdof), eveci(1:NGdof, 1:NGdof), revecr(1:NGdof, 1:2 * NGdof), evecl(1:NGdof, 1:NGdof)
    real(rkind) :: work(1:4 * NGdof)! for construction of evalues/evectors
    character :: JOB

    integer :: iev

    real(rkind) :: dRZ != 1.0e-03

    real(rkind) :: lmu(1:Mvol), lpflux(1:Mvol), lhelicity(1:Mvol)! original profiles

    integer :: idgetrf, ipiv(1:Ngdof)
    real(rkind) :: sgn

    ! Only makes sense to compute the Hessian if helicity is constrained
    if (Lconstraint /= 2) then
      write(*, *)'hesian: only calculates the hessian matrix when Lcons=2'
      return
    end if

    lmu(1:Nvol) = mu(1:Nvol)
    lpflux(1:Nvol) = pflux(1:Nvol)
    lhelicity(1:Nvol) = helicity(1:Nvol)! save original profile information

    oldEnergy(0) = Energy ! Energy was calculated in dforce

    xx(0, -2:2) = 0.0_rkind

    if (LHmatrix .and. Igeometry == 2) then
      call sallocate(HdFFdRZ, 1, LGdof, 0, 1, 1, LGdof, 0, 1, 1, Mvol, 0.0_rkind)
    end if
    call sallocate(dBBdmp, 1, LGdof, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)
    call sallocate(denergydrr, 1, LGdof, 1, Mvol, 0, 1, 1, LGdof, 0, 1, 0.0_rkind)
    call sallocate(denergydzr, 1, LGdof, 1, Mvol, 0, 1, 1, LGdof, 0, 1, 0.0_rkind)

    if (LocalConstraint) then
      call sallocate(dmupfdx, 1, Mvol, 1, 1, 1, 2, 1, LGdof, 0, 1, 0.0_rkind)
    else
      call sallocate(dmupfdx, 1, Mvol, 1, Mvol - 1, 1, 2, 1, LGdof, 0, 1, 0.0_rkind)
    end if

    call sallocate(hessian2D, 1, NGdof, 1, NGdof, 0.0_rkind)

    !if (LHmatrix) then
    Lhessian2Dallocated = .true.
    !else
    force_ft_trunc_grad_alloc = .true.
    !endif
    !This step cleared.

    LComputeDerivatives = .true.
    LComputeAxis = .false.
    call dforce(NGdof, position(0:NGdof), force(0:NGdof), LComputeDerivatives, LComputeAxis)! calculate force-imbalance
    ! & hessian

    ohessian(1:NGdof, 1:NGdof) = hessian2D(1:NGdof, 1:NGdof)! internal copy

    if (Lcheck == 5) then ! check construction of Hessian

      xx(0, -2:2) = 0.0_rkind
      dRZ = 1.0e-04_rkind

      write(svol, '(i3.3)')myid

      tdof = 0 ! geometrical degree-of-freedom counter labels derivative

      do vvol = 1, Mvol - 1 ! loop over internal interfaces

        cput = MPI_Wtime()

        do ii = 1, mn ! loop over Fourier harmonics degrees of freedom

          do irz = 0, 1 ! loop over R,Z degrees of freedom

            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

            do issym = 0, 1

              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics

              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! this is not a degree of freedom: no dependence on
              ! Zbs_{m,n} for m=0, n=0
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! this is not a degree of freedom: no dependence on
              ! Rbs_{m,n} for m=0, n=0

              tdof = tdof + 1

              do isymdiff = -2, 2

                if (isymdiff == 0) cycle

                xx(1:NGdof, isymdiff) = position(1:NGdof)! reset geometry to original

                xx(tdof, isymdiff) = position(tdof) + isymdiff * dRZ ! perturb appropriate geometric harmonic

                LComputeDerivatives = .false.
                LComputeAxis = .true.
                call dforce(NGdof, xx(0:NGdof, isymdiff), ff(0:NGdof, isymdiff), LComputeDerivatives, LComputeAxis)!
                !force-imbalance

              end do

              df(1:NGdof) = (-ff(1:NGdof, +2) + 8 * ff(1:NGdof, +1) - 8 * ff(1:NGdof, -1) + ff(1:NGdof, -2)) / (12 * dRZ)

              tdoc = 0

              do jvol = 1, Mvol - 1

                do jj = 1, mn

                  do jrz = 0, 1 ! loop over R,Z degrees of freedom

                    if (jrz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

                    do jssym = 0, 1

                      if (jssym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics

                      if (jj == 1 .and. jrz == 1 .and. jssym == 0) cycle ! this is not a constraint: I_{m,n} for m=0, n=0
                      ! is irrelevant
                      if (jj == 1 .and. jrz == 0 .and. jssym == 1) cycle ! this is not a constraint: I_{m,n} for m=0, n=0
                      ! is irrelevant

                      tdoc = tdoc + 1

                      call fatal(tdoc < 1, __FILE__, "needs attention ")

                      call fatal(tdoc > NGdof, __FILE__, "needs attention ")

                      cput = MPI_Wtime()

                      error = abs(df(tdoc) - force_ft_trunc_grad(tdoc, tdof))

                      if (abs(force_ft_trunc_grad(tdoc, tdof)) > 1.0e-05_rkind .or. abs(df(tdoc)) > 1.0e-05_rkind .or. error > dRZ)&
                          & then ! write to screen
                        write(stdout, 1001)cput - cpus, myid, vvol, im(ii), in(ii), irz, issym, tdof, jvol, im(jj),&
                            & in(jj), jrz, jssym, tdoc, df(tdoc), tdoc, tdof, force_ft_trunc_grad(tdoc, tdof), error
                      end if
                    end do

                  end do
                end do

              end do

              1001 format("hesian : ", f10.2, " : myid=", i3, " ; vvol=", i3, " ; (", i4, " ,", i4, " ); irz=", i2, " ; issym=", &
                  i2, " tdof=", i6,  " ; jvol=", i4, " ; (", i4, " ,", i4, " ); jrz=", i2, " ; jssym=", i2, " tdoc=", i6, &
                  " ; fd-estimate=", es13.5, "                 hessian(", i6, ",", i6, " )=", es13.5, " ;", :, " err=", es13.5, &
                  " ;", :, f12.4, " ")
            end do

          end do

        end do

      end do

      pack = 'U' ! position(0) = 0.0_rkind ! this is not used
      call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol),&
          & iZbc(1:mn, 0:Mvol), pack, .false., LComputeAxis)

      mu(1:Nvol) = lmu(1:Nvol)
      pflux(1:Nvol) = lpflux(1:Nvol)
      helicity(1:Nvol) = lhelicity(1:Nvol)

    end if

    if ((LHevalues .or. LHevectors)) then

      if (myid == 0) then
        cput = MPI_Wtime()
        write(stdout, '("hesian : ",f10.2," : LHevalues="L2" , LHevectors="L2" ")')cput - cpus, LHevalues, LHevectors
      end if

      evalr(1:NGdof) = 0.0_rkind
      evecr(1:NGdof, 1:NGdof) = 0.0_rkind
      evali(1:NGdof) = 0.0_rkind
      eveci(1:NGdof, 1:NGdof) = 0.0_rkind

      if (LHevectors) then
        JOB = 'V'
        Ldvr = NGdof
        Ldvi = NGdof
      else
        JOB = 'N'
        Ldvr = 1
        Ldvi = 1 ! provide dummy values when eigenvectors are not required
      end if

      cpul = MPI_Wtime()

      if02ebf = 1
      LDA = NGdof
      Lwork = 4 * NGdof

      hessian2D(1:NGdof, 1:NGdof) = ohessian(1:NGdof, 1:NGdof)

      call dgeev('N', JOB, NGdof, hessian2D(1:LDA, 1:NGdof), LDA, evalr(1:NGdof), evali(1:NGdof), evecl(1:Ldvr, 1:NGdof),&
          & Ldvr, revecr(1:Ldvr, 1:2 * NGdof), Ldvr, work(1:Lwork), Lwork, if02ebf)
      evecr(1:Ldvr, 1:NGdof) = revecr(1:Ldvr, 1:NGdof)
      eveci(1:Ldvr, 1:NGdof) = revecr(1:Ldvr, NGdof + 1:2 * NGdof)

      if (myid == 0) then
        cput = MPI_Wtime()
        if (if02ebf < 0) then
          write(stdout, '("hesian : ",f10.2," : DGEEV error the "i2" th argument had illegal value  ;   time="f10.2"s    &
              &         ")')cput - cpus, -if02ebf, cput - cpul
        else if (if02ebf > 0) then
          write(stdout, '("hesian : ",f10.2," : DGEEV error, factorization failed  ;   time="f10.2"s ")')cput - cpus,&
              & cput - cpul
        else
          write(stdout, '("hesian : ",f10.2," : computed evalues ; if02ebf="i2" ; success ;       time="f10.2"s ")')cput&
              & - cpus, if02ebf, cput - cpul
        end if
      end if

      do lvol = 1, Mvol - 1
        do ii = 1, mn
          evecr(ii + (lvol - 1) * mn, 1:NGdof) = evecr(ii + (lvol - 1) * mn, 1:NGdof) * psifactor(ii, lvol)! geometrical
          ! regularization
        end do
      end do

      if (myid == 0) then ! screen output

        if (LHevectors) then

          do iev = 1, NGdof ! loop over all eigenvalues
            if (evalr(iev) < 0.0_rkind) then ! only show unstable eigenvalues
              write(stdout, '("hesian : ",f10.2," : evalr="es13.5"  ")')cput - cpus, evalr(iev)
              write(stdout, '("hesian : ",es10.3," : "999(" ("i3","i3")":))')evalr(iev), [(im(ii), in(ii), ii = 1, mn)]
              do lvol = 1, Mvol - 1
                write(stdout, '("hesian : ",i10   " : "999es10.2)')lvol, evecr(1 + mn * (lvol - 1) :mn + mn * (lvol - 1) ,&
                    & iev)
              end do
            end if
          end do

        else ! matches if( LHvectors)

          do iev = 1, NGdof
            write(stdout, '("hesian : ",f10.2," : ",i6," : evalue=(",es23.15," ,",es23.15," ) ")')cput - cpus, iev,&
                & evalr(iev), evali(iev)
            !endif
          end do
        end if
      end if
    end if

    ! output hessian, eigenvalues, and eigenvectors to the .h5 file
    if (LHmatrix .and. Lconstraint == 2) then
      call write_stability(ohessian, evalr, evali, evecr, NGdof)
    end if

    if (myid == 0) then

      hessian2D(1:NGdof, 1:NGdof) = ohessian(1:NGdof, 1:NGdof)

      call dgetrf(NGdof, NGdof, hessian2D(1:NGdof, 1:NGdof), NGdof, ipiv(1:NGdof), idgetrf)

      determinant = 1.0_rkind

      do iev = 1, NGdof
        determinant = determinant * hessian2D(iev, iev)!calculate determinant from factorized form of hessian
      end do

      sgn = 1.0_rkind

      do iev = 1, NGdof
        if (ipiv(iev)/= iev) then
          sgn = -sgn
        end if
      end do

      determinant = sgn * determinant !correct for the sign of the determinant

      select case (idgetrf)
      case (0)
        write(stdout, '("hesian : " 10x " : myid="i3" ; idgetrf="i3" ;             ; determinant="es13.5" ")')myid,&
            & idgetrf, determinant
      case (1:)
        write(stdout, '("hesian : " 10x " : myid="i3" ; idgetrf="i3" ; singular    ; determinant="es13.5" ")')myid,&
            & idgetrf, determinant
      case (:-1)
        write(stdout, '("hesian : " 10x " : myid="i3" ; idgetrf="i3" ; input error ; determinant="es13.5" ")')myid,&
            & idgetrf, determinant
      case default
        call fatal(.true., __FILE__, "illegal ifail returned from dgetrf ")

      end select

    end if

    if (LHmatrix .and. Igeometry == 2) then
      deallocate(HdFFdRZ)
    end if

    deallocate(dBBdmp)
    deallocate(dmupfdx)
    deallocate(denergydrr)
    deallocate(denergydzr)

    deallocate(hessian2D)
    write(stdout, *)5656

  end subroutine compute_hessian
end module hessian_mod
