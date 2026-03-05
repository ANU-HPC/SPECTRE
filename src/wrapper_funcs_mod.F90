module wrapper_funcs_mod
#include "mpi_precomp.h"
  USE_MPI
  use omp_lib

  use, intrinsic :: ieee_arithmetic

  use allglobal_mod, only : myid, cpus, MPI_COMM_SPEC, ext, ivol, im, in, Mvol, Lcoordinatesingularity, Lplasmaregion, &
      Lvacuumregion, cheby, zernike, Ate, Aze, Ato, Azo, sg, inv_jac, guvij, Rij, Zij, Nt, Nz, efmn, ofmn, cfmn, sfmn, &
      Node, pi2nfp, dtflux, dpflux, wrtend, ncpu, mn, Ntz, LGdof, NGdof, iRbc, iZbs, iRbs, iZbc, &
      ImagneticOK, BnsErr, iBns, iBnc, iVns, iVnc, nfreeboundaryiterations, first_free_bound, solution, &
      IPDt, LocalConstraint, dBBdmp, dmupfdx, dRodR, dRodZ, dZodR, dZodZ, dRadR, dRadZ, dZadR, dZadZ, &
      ijreal, ijimag, jireal, Iquad, gaussianabscissae, &
      LcoordinateSingularity, iRij, iZij, dRij, dZij, tRij, tZij, mmpp, NAdof, mns, dBdX, sweight, diotadxup, dItGpdxtp, &
      psifactor, IsMyVolume, IsMyVolumeValue, WhichCpuID, cosi, sini, Bemn, &
      Bomn, Iomn, Iemn, Somn, Semn, lsavedguvij, guvijsave, Tss, Dtc, Dzc, Ttc, Tzc, Tsc, Dts, Dzs, Tts, Tzs, &
      build_vector_potential, vvolume, xoffset, IPdtdPf, dvolume, force_ft_trunc_grad, Romn_ext, &
      Remn_ext, Zomn_ext, Zemn_ext, jiimag, jkimag, jkreal, kjimag, kjreal
  use base_functions_mod, only: get_cheby_d2, get_zernike_d2
  use constants_mod, only : pi2, mu0
  use errors_mod, only: fatal
  use forces_mod, only: dforce, dfp100, lforce, get_perturbed_solution, evaluate_dmupfdx, get_LU_beltrami_matrices
  use geometry_mod, only: compute_coordinates
  use intghs_mod, only: wk, intghs_workspace_init, intghs_workspace_destroy
  use io_mod, only : stdout, lunit
  use input_list_mod, only : Nfp, Igeometry, Nvol, Lrad, tflux, pflux, phiedge, pressure, pscale, helicity, Ladiabatic, &
      adiabatic, gamma, Rbc, Zbs, Rbs, Zbc, Lconstraint, Lfreebound, Lbnszero, mfreeits, gBntol, gBnbld, vcasingtol, LautoinitBn, &
      mmRZRZ, nnRZRZ, allRZRZ, &
      odetol, nPpts, nPtrj, LHevalues, LHevectors, LHmatrix, Lcheck, Lzerovac, mu, Isurf, Ivolume, Mpol, &
      enforce_stell_sym, Ntor, rpol, rtor, wpoloidal, Ngrid, Bns, Bnc, Rac, Zas, Ras, Zac
  use magnetic_field_mod, only: compute_magnetic_field
  use freeb_mod, only: compute_bnorm_plasmab
  use memory_allocation_mod, only: sallocate, nallocate
  use memory_mod, only: allocate_Beltrami_matrices, &
      allocate_geometry_matrices, &
      deallocate_Beltrami_matrices, &
      deallocate_geometry_matrices
  use mpi_collectives_mod, only: broadcast
  use numerical_mod, only: invfft, set_fftw_plan
  use packing_mod, only: packxi, packab
  use precisions_mod, only: rkind, small
  use typedefns_mod, only: MatrixLU
  use volume_mod, only: lregion, volume
  use coordinate_mod, only: calc_axis_deriv_grid
  use field_mod, only: alloc_ntz_arrays

  implicit none

  private
  public :: get_angle_constraint, angconst_dfp200, alloc_for_hessian
  public :: dealloc_hessian, alloc_remn_ext, get_mesh, update_bnorm, calc_beltrami_error
  public :: get_vec_pot, get_grid_data, print_threads, print_nans, alloc_allrzrz, prepare_endfile_writing

  real(rkind), allocatable :: freal_jac(:, :, :, :, :)
  real(rkind), allocatable :: freal_dBBdmp(:, :, :, :)

contains

  subroutine print_nans(a)
    use, intrinsic :: ieee_arithmetic
    implicit none
    real(rkind), intent(in) :: a(:)
    integer :: i
    logical :: any_nan

    any_nan = .false.

    do i = 1, size(a)
        if (ieee_is_nan(a(i))) then
            print *, "NaN at index:", i
            any_nan = .true.
        end if
    end do

    if (.not. any_nan) print *, "No NaNs found."
  end subroutine print_nans

  subroutine get_jac(jac, lvol, lss, Lcurvature, Ntz, mn)
    integer, intent(in) :: lvol, Lcurvature, Ntz, mn
    real(rkind), intent(in) :: lss

    integer :: ii, jj, kk, irz, innout, issym, signlss, mi, ni, imn
    real(rkind) :: Remn(1:mn, 0:2), Zomn(1:mn, 0:2), Romn(1:mn, 0:2), Zemn(1:mn, 0:2), alss, blss, sbar, sbarhim(1:mn), fj(1:mn, 0:&
        2)
    real(rkind), intent(out) :: jac(1:Ntz)
    ! real(rkind), intent(out)   :: metric(1:Ntz, 1:3, 1:3)

    jac(1:Ntz) = 0.0_rkind
    ! metric(1:Ntz, 1:3, 1:3) = 0.0_rkind

    Rij(1:Ntz, 0:3, 0:3) = 0.0_rkind
    Zij(1:Ntz, 0:3, 0:3) = 0.0_rkind

    Remn(1:mn, 0:2) = 0.0_rkind ! interpolated coordinate harmonics; 6 Feb 13
    Zomn(1:mn, 0:2) = 0.0_rkind
    Romn(1:mn, 0:2) = 0.0_rkind
    Zemn(1:mn, 0:2) = 0.0_rkind

    if (Lcoordinatesingularity) then

      sbar = (lss + 1.0_rkind) * 0.5_rkind

      select case (Igeometry)
      case (2)
        fj(1:Ntor + 1, 0) = sbar ! these are the mj==0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
        fj(Ntor + 2:mn, 0) = sbar ** (im(Ntor + 2:mn) + 1) ! these are the me.ne.0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
      case (3)
        fj(1:Ntor + 1, 0) = sbar ** 2 ! switch to sbar=r; 29 Jun 19
        fj(Ntor + 2:mn, 0) = sbar ** im(Ntor + 2:mn)! these are the me.ne.0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
      case default
        call fatal(.true., __FILE__, "invalid Igeometry for Lcoordinatesingularity=T")
      end select

      Remn(1:mn, 0) = iRbc(1:mn, 0) + (iRbc(1:mn, 1) - iRbc(1:mn, 0)) * fj(1:mn, 0)
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 0) = iRbs(1:mn, 0) + (iRbs(1:mn, 1) - iRbs(1:mn, 0)) * fj(1:mn, 0)
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 0) = iZbs(1:mn, 0) + (iZbs(1:mn, 1) - iZbs(1:mn, 0)) * fj(1:mn, 0)
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 0) = iZbc(1:mn, 0) + (iZbc(1:mn, 1) - iZbc(1:mn, 0)) * fj(1:mn, 0)
        end if
      end if

    else ! matches if( Lcoordinatesingularity ) ; 22 Apr 13

      alss = 0.5_rkind * (1.0_rkind - lss)
      blss = 0.5_rkind * (1.0_rkind + lss)

      Remn(1:mn, 0) = alss * iRbc(1:mn, lvol - 1) + blss * iRbc(1:mn, lvol)
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 0) = alss * iRbs(1:mn, lvol - 1) + blss * iRbs(1:mn, lvol)
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 0) = alss * iZbs(1:mn, lvol - 1) + blss * iZbs(1:mn, lvol)
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 0) = alss * iZbc(1:mn, lvol - 1) + blss * iZbc(1:mn, lvol)
        end if
      end if ! end of if( Igeometry==3 ) ; 01 Feb 13

    end if ! end of if( Lcoordinatesingularity ); 01 Feb 13

    call invfft(mn, im(1:mn), in(1:mn), Remn(1:mn, 0), Romn(1:mn, 0), Zemn(1:mn, 0), Zomn(1:mn, 0), &
        Rij(1:Ntz, 0, 0), Zij(1:Ntz, 0, 0))! maps to real(rkind) space

    if (Lcoordinatesingularity) then

      select case (Igeometry)
      case (2)
        fj(1:Ntor + 1, 1) = 0.5_rkind ! these are the mj==0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
        fj(Ntor + 2:mn, 1) = 0.5_rkind * (im(Ntor + 2:mn) + 1.0_rkind) * fj(Ntor + 2:mn, 0) / sbar ! these are the me.ne.0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
      case (3)
        fj(1:Ntor + 1, 1) = sbar ! these are the mj==0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
        fj(Ntor + 2:mn, 1) = 0.5_rkind * im(Ntor + 2:mn) * fj(Ntor + 2:mn, 0) / sbar ! these are the me.ne.0 harmonics; 11 Aug 14; switch to sbar=r; 29 Jun 19
      case default
        call fatal(.true., __FILE__, "invalid Igeometry for Lcoordinatesingularity=T and Lcurvature .ne. 0")
      end select

      Remn(1:mn, 1) = (iRbc(1:mn, 1) - iRbc(1:mn, 0)) * fj(1:mn, 1)
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 1) = (iRbs(1:mn, 1) - iRbs(1:mn, 0)) * fj(1:mn, 1)
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 1) = (iZbs(1:mn, 1) - iZbs(1:mn, 0)) * fj(1:mn, 1)
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 1) = (iZbc(1:mn, 1) - iZbc(1:mn, 0)) * fj(1:mn, 1)
        end if
      end if

    else ! matches if( Lcoordinatesingularity ) ; 22 Apr 13

      Remn(1:mn, 1) = (-iRbc(1:mn, lvol - 1) + iRbc(1:mn, lvol)) * 0.5_rkind
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 1) = (-iRbs(1:mn, lvol - 1) + iRbs(1:mn, lvol)) * 0.5_rkind
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 1) = (-iZbs(1:mn, lvol - 1) + iZbs(1:mn, lvol)) * 0.5_rkind
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 1) = (-iZbc(1:mn, lvol - 1) + iZbc(1:mn, lvol)) * 0.5_rkind
        end if
      end if ! end of if( Igeometry==3 ) ; 01 Feb 13

    end if ! end of if( Lcoordinatesingularity ); 01 Feb 13

    ! Derivative w.r.t s
    call invfft(mn, im(1:mn), in(1:mn), Remn(1:mn, 1), Romn(1:mn, 1), Zemn(1:mn, 1), Zomn(1:mn, 1), &
        Rij(1:Ntz, 1, 0), Zij(1:Ntz, 1, 0))! maps to real(rkind) space

    ! Derivative w.r.t theta
    call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * Romn(1:mn, 0), -im(1:mn) * Remn(1:mn, 0), im(1:mn) * Zomn(1:mn, 0), -im(1:mn) * &
        Zemn(1:mn, 0), &
        Rij(1:Ntz, 2, 0), Zij(1:Ntz, 2, 0))! maps to real(rkind) space

    ! Derivative w.r.t zeta
    call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * Romn(1:mn, 0), in(1:mn) * Remn(1:mn, 0), -in(1:mn) * Zomn(1:mn, 0), in(1:mn) * &
        Zemn(1:mn, 0), &
        Rij(1:Ntz, 3, 0), Zij(1:Ntz, 3, 0))! maps to real(rkind) space

    do ii = 1, 3
      Rij(1:Ntz, 0, ii) = Rij(1:Ntz, ii, 0)! just to complete workspace arrays; 22 Apr 13
      Zij(1:Ntz, 0, ii) = Zij(1:Ntz, ii, 0)! just to complete workspace arrays; 22 Apr 13
    end do

    select case (Igeometry)

    case (1) ! Igeometry=1; Cartesian

      jac(1:Ntz) = Rij(1:Ntz, 1, 0) * rpol * rtor

      ! do ii = 1, 3
      !     do jj = ii, 3; metric(1:Ntz, ii, jj) = Rij(1:Ntz, ii, 0)*Rij(1:Ntz, jj, 0)
      !     end do
      ! end do
      ! metric(1:Ntz, 2, 2) = metric(1:Ntz, 2, 2) + rpol*rpol
      ! metric(1:Ntz, 3, 3) = metric(1:Ntz, 3, 3) + rtor*rtor

    case (2) ! Igeometry=2; cylindrical

      jac(1:Ntz) = Rij(1:Ntz, 1, 0) * Rij(1:Ntz, 0, 0)

      ! do ii = 1, 3
      !     do jj = ii, 3; metric(1:Ntz, ii, jj, 0) = Rij(1:Ntz, ii, 0)*Rij(1:Ntz, jj, 0)
      !     end do
      ! end do
      ! metric(1:Ntz, 2, 2) = metric(1:Ntz, 2, 2) + Rij(1:Ntz, 0, 0)*Rij(1:Ntz, 0, 0)
      ! metric(1:Ntz, 3, 3) = metric(1:Ntz, 3, 3) + 1.0_rkind

    case (3) ! Igeometry=3; toroidal

      jac(1:Ntz) = Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, 0))

      ! do ii = 1, 3
      !     do jj = ii, 3; metric(1:Ntz, ii, jj) = Rij(1:Ntz, ii, 0)*Rij(1:Ntz, jj, 0) + Zij(1:Ntz, ii, 0)*Zij(1:Ntz, jj, 0)
      !     end do
      ! end do
      ! metric(1:Ntz, 3, 3) = metric(1:Ntz, 3, 3) + (Rij(1:Ntz, 0, 0))**2

    end select ! matches select case( Lcurvature ) ; 10 Mar 13
  end subroutine get_jac

  subroutine get_angle_constraint(IIl_total, dLL_total)
    integer :: iocons
    real(rkind) :: XX(1:Ntz), YY(1:Ntz), DDl, MMl
    real(rkind) :: length(1:Ntz)
    integer :: Lcurvature, ii, jj, kk, ll, ifail, ivol, lnn, mi, vvol
    real(rkind) :: lss
    real(rkind), intent(out) :: IIl_total(1:Mvol, 1:Ntz), dLL_total(1:Mvol, 0:1, 1:Ntz)

    do vvol = 1, Mvol

      do iocons = 0, 1

        lss = 2.0_rkind * iocons - 1.0_rkind ! recall that iocons is effective local radial coordinate 24 Apr 13

        Lcurvature = 1

        call compute_coordinates(vvol, lss, Lcurvature, Nt, Nz, mn)! get coordinates and derivatives wrt Rj, Zj, at specific radial location

        do ivol = 0, 1
          call invfft(mn, im(1:mn), in(1:mn), iRbc(1:mn, vvol - 1 + ivol), iRbs(1:mn, vvol - 1 + ivol), &
              iZbc(1:mn, vvol - 1 + ivol), iZbs(1:mn, vvol - 1 + ivol), &
              iRij(1:Ntz, vvol - 1 + ivol), iZij(1:Ntz, vvol - 1 + ivol))

          call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * iRbs(1:mn, vvol - 1 + ivol), -im(1:mn) * iRbc(1:mn, vvol - 1 + ivol), &
              im(1:mn) * iZbs(1:mn, vvol - 1 + ivol), -im(1:mn) * iZbc(1:mn, vvol - 1 + ivol), &
              tRij(1:Ntz, vvol - 1 + ivol), tZij(1:Ntz, vvol - 1 + ivol))
        end do ! end of do ivol = 0, 1  18 Jul 14

        dRij(1:Ntz, vvol) = iRij(1:Ntz, vvol) - iRij(1:Ntz, vvol - 1)
        dZij(1:Ntz, vvol) = iZij(1:Ntz, vvol) - iZij(1:Ntz, vvol - 1)

        length(1:Ntz) = sqrt(dRij(1:Ntz, vvol) ** 2 + dZij(1:Ntz, vvol) ** 2)

        dLL_total(vvol, iocons, 1:Ntz) = (dRij(1:Ntz, vvol) * tRij(1:Ntz, vvol - 1 + iocons) &
            + dZij(1:Ntz, vvol) * tZij(1:Ntz, vvol - 1 + iocons)) / length(1:Ntz)

        if (iocons == 1) then ! include spectral condensation constraints local to interface, i.e. no tri-diagonal structure
          efmn(1:mn) = (mmpp(1:mn)) * iRbc(1:mn, vvol)
          sfmn(1:mn) = (mmpp(1:mn)) * iZbs(1:mn, vvol)

          if (.not.enforce_stell_sym) then
            ofmn(1:mn) = (mmpp(1:mn)) * iRbs(1:mn, vvol)
            cfmn(1:mn) = (mmpp(1:mn)) * iZbc(1:mn, vvol)
          else
            ofmn(1:mn) = 0.0_rkind
            cfmn(1:mn) = 0.0_rkind
          end if

          call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), &
              XX(1:Ntz), YY(1:Ntz))

          if (enforce_stell_sym) then
            DDl = sum((iRbc(1:mn, vvol) ** 2 + iZbs(1:mn, vvol) ** 2))
            MMl = sum(mmpp(1:mn) * (iRbc(1:mn, vvol) ** 2 + iZbs(1:mn, vvol) ** 2) ) / DDl
          else
            DDl = sum((iRbc(1:mn, vvol) ** 2 + iZbs(1:mn, vvol) ** 2 + iRbs(1:mn, vvol) ** 2 + iZbc(1:mn, vvol) ** 2))
            MMl = sum(mmpp(1:mn) * (iRbc(1:mn, vvol) ** 2 + iZbs(1:mn, vvol) ** 2 + iRbs(1:mn, vvol) ** 2 + iZbc(1:mn, vvol)** 2&
                &) ) / DDl
          end if

          IIl_total(vvol, 1:Ntz) = tRij(1:Ntz, vvol) * (XX(1:Ntz) - MMl * iRij(1:Ntz, vvol)) &
              + tZij(1:Ntz, vvol) * (YY(1:Ntz) - MMl * iZij(1:Ntz, vvol))

        end if
      end do
    end do
  end subroutine get_angle_constraint

  subroutine angconst_dfp200(ang_constr_jac_full, add_psifactor, add_sweightfactor)
    logical :: LInnerVolume
    logical, intent(in) :: add_psifactor, add_sweightfactor

    integer :: NN, IA, ifail, if01adf, vflag, MM, idgetrf, idgetri, lvol, pvol
    integer :: vvol, innout, ii, jj, irz, issym, iocons, idof, imn, ll
    integer :: Lcurvature, ideriv, id, ierr
    integer :: iflag, cpu_id, cpu_id1, even_or_odd, vol_parity
    integer :: tag, tag2, req1, req2, req3, req4
    MPI_STATUS_TYPE :: stat

    real(rkind) :: lastcpu, lss, lfactor, DDl, MMl
    real(rkind), allocatable :: XX(:), YY(:), dBB(:, :), dII(:), dLL(:), length(:), constraint(:)

    real(rkind), intent(inout) :: ang_constr_jac_full(0:1, 1:LGdof * (Mvol - 1) , 1:(Mvol - 1), 1:Ntz)
    real(rkind) :: ang_constr_jac(0:1, 1:Ntz, 0:1, 1:LGdof, 0:1, 1:Mvol)

    integer :: tdof, ivol

    ang_constr_jac = 0.0_rkind

    call sallocate(dBB, 1, Ntz, -1, 2, 0.0_rkind)! magnetic field strength (on interfaces) in real(rkind) space and derivatives
    call sallocate(XX, 1, Ntz, 0.0_rkind)
    call sallocate(YY, 1, Ntz, 0.0_rkind)
    call sallocate(length, 1, Ntz, 0.0_rkind)! this is calculated in lforce
    call sallocate(dII, 1, Ntz, 0.0_rkind)! spectral constraint
    call sallocate(dLL, 1, Ntz, 0.0_rkind)! length   constraint

    if (LocalConstraint) then

      do vvol = 1, Mvol

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

        dBdX%vol = vvol ! Label
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz)! compute volume

        idof = 0 ! labels degree of freedom; local to interface

        do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol

          do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii

            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

            do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms

              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

              idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

              do innout = 0, 1 ! loop over deformations to inner and outer interface; inside do vvol; inside do ii; inside do irz

                if (vvol == 1 .and. innout == 0) cycle ! no degrees of freedom at coordinate axis / fixed inner boundary
                if (vvol == Mvol .and. innout == 1) cycle ! no degress of freedom                      fixed outer boundary; for linearized displacement

                call angconst_evaluate_dBB(ang_constr_jac, vvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dII, dLL, Ntz&
                    , add_psifactor, add_sweightfactor)
              end do ! matches do innout
            end do ! matches do issym
          end do ! matches do irz
        end do ! matches do ii
      end do ! matches do vvol = 1, Mvol

    else ! CASE SEMI GLOBAL CONSTRAINT

      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz)! compute volume
      end do

      do even_or_odd = 0, 1 ! First loop on even interfaces perturbation, then on odd interfaces. This allow efficient parallelization

        do vvol = 1, Mvol - 1 !labels which interface is perturbed

          ! Parallelization - parallelization on perturbed interface and not on volume (outermost loop)
          vol_parity = MODULO(vvol, 2)
          if ( (vol_parity == 0) .and. (even_or_odd == 1) ) cycle ! even_or_odd=1 thus perturb only odd  interfaces
          if ( (vol_parity == 1) .and. (even_or_odd == 0) ) cycle ! even_or_odd=0 thus perturb only even interfaces

          call IsMyVolume(vvol)

          if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's inner volume

            call IsMyVolume(vvol + 1)

            if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's outer volume either - cycle
              cycle
            else if (IsMyVolumeValue == -1) then
              call fatal(.true., __FILE__, "Unassociated volume")
            else
              LInnerVolume = .false.
            end if

          else if (IsMyVolumeValue == -1) then
            call fatal(.true., __FILE__, "Unassociated volume")
          else
            LinnerVolume = .true.
          end if

          idof = 0 ! labels degree of freedom of interface vvol

          do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol

            do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii
              if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

              do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

                idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

                do lvol = vvol, vvol + 1

                  ! Only interested in one interface perturbation
                  if (ncpu > 1) then
                    if (LinnerVolume .and. (lvol == vvol + 1) ) cycle
                    if (.not.LinnerVolume .and. (lvol == vvol) ) cycle
                  end if

                  ! Set up perturbation information
                  if (lvol == vvol) innout = 1 ! Perturb w.r.t outer interface
                  if (lvol == vvol + 1) innout = 0 ! Perturb w.r.t inner interface

                  ! Set up volume information
                  call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.
                  NN = NAdof(lvol)

                end do ! end of do lvol = vvol, vvol+1

                ! Now volumes neighbouring the interface share perturbed solution
                cpu_id = WhichCpuID(vvol, ncpu)
                cpu_id1 = WhichCpuID(vvol + 1, ncpu)

                do lvol = vvol, vvol + 1
                  call IsMyVolume(lvol)

                  if (IsMyVolumeValue == 0) then
                    cycle
                  else if (IsMyVolumeValue == -1) then
                    call fatal(.true., __FILE__, "Unassociated volume")
                  end if

                  if (lvol == vvol) then
                    innout = 1
                  else
                    innout = 0
                  end if

                  call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

                  call angconst_evaluate_dBB(ang_constr_jac, lvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dII, dLL, Ntz&
                      , add_psifactor, add_sweightfactor)

                end do ! matches do lvol = vvol, vvol+1

              end do ! matches do issym
            end do ! matches do irz
          end do ! matches do ii
        end do ! matches do vvol
      end do ! matches do even_or_odd
    end if ! End of if( LocalConstraint )

    do vvol = 1, Mvol
      cpu_id = WhichCpuID(vvol, ncpu)
      call MPI_BCAST(ang_constr_jac(0:1, 1:Ntz, 0:1, 1:LGdof, 0:1, vvol), 2 * 2 * 2 * LGdof * Ntz, MPI_DOUBLE_PRECISION, cpu_id, &
          MPI_COMM_SPEC, ierr)
    end do

    do vvol = 1, Mvol - 1 ! loop over interior surfaces

      if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this interface are valid

        idof = 0 ! labels degree-of-freedom = Fourier harmonic of surface geometry

        do ii = 1, mn ! loop over degrees-of-freedom

          do irz = 0, 1 ! Fourier harmonic of R, Fourier harmonic of Z

            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

            do issym = 0, 1 ! stellarator symmetry

              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on the non-stellarator symmetric harmonics

              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

              idof = idof + 1 ! labels degree-of-freedom

              do ivol = 1, Mvol - 1
                tdof = (vvol - 1) * LGdof + idof
                if (ivol == vvol - 1) then
                  ang_constr_jac_full(0:1, tdof, ivol, 1:Ntz) = ang_constr_jac(0:1, 1:Ntz, 0, idof, 1, ivol + 1)
                elseif (ivol == vvol) then
                  ang_constr_jac_full(0:1, tdof, ivol, 1:Ntz) = ang_constr_jac(0:1, 1:Ntz, 0, idof, 0, ivol + 1) &
                      - ang_constr_jac(0:1, 1:Ntz, 1, idof, 1, ivol)
                elseif (ivol == vvol + 1) then
                  ang_constr_jac_full(0:1, tdof, ivol, 1:Ntz) = -ang_constr_jac(0:1, 1:Ntz, 1, idof, 0, ivol)
                end if
              end do
            end do
          end do
        end do
      end if
    end do

    deallocate(dLL)
    deallocate(dII)
    deallocate(dBB)
    deallocate(XX)! spectral constraints; not used
    deallocate(YY)
    deallocate(length)
  end subroutine angconst_dfp200

  subroutine angconst_evaluate_dBB(ang_constr_jac, lvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dII, dLL, Ntz, &
      add_psifactor, add_sweightfactor)
    logical, intent(in) :: add_psifactor, add_sweightfactor
    integer :: iocons, lvol, ideriv, id, iflag, Lcurvature, innout, issym, irz, ii, ifail, idof, Ntz
    real(rkind) :: DDl, MMl, fact, sfact
    real(rkind) :: XX(1:Ntz), YY(1:Ntz), dBB(1:Ntz, -1:2), dII(1:Ntz), dLL(1:Ntz)
    real(rkind) :: length(1:Ntz), constraint(1:Ntz)

    real(rkind), intent(inout) :: ang_constr_jac(0:1, 1:Ntz, 0:1, 1:LGdof, 0:1, 1:Mvol)

    do iocons = 0, 1

      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis); no constraints
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; no constraints

      ideriv = -1
      iflag = 0

      call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)

      dLL(1:Ntz) = 0.0_rkind ! either no spectral constraint, or not the appropriate interface

      if (Igeometry >= 3) then ! spectral constraints are only required in toroidal or extended-cylindrical geometry

        if (innout == 1 .and. iocons == 1) then ! include derivatives of spectral constraints

          if (issym == 0) then ! take derivatives wrt Rbc and Zbs
            if (irz == 0) then ! take derivative wrt Rbc
              dII(1:Ntz) = -im(ii) * sini(1:Ntz, ii) * (XX(1:Ntz) - MMl * iRij(1:Ntz, lvol)) &
                  - 2.0_rkind * (mmpp(ii) - MMl) * iRbc(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * &
                  iZij(1:Ntz, lvol)) / DDl &
                  + Rij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * cosi(1:Ntz, ii)
            else ! take derivative wrt Zbs
              dII(1:Ntz) = +im(ii) * cosi(1:Ntz, ii) * (YY(1:Ntz) - MMl * iZij(1:Ntz, lvol)) &
                  - 2.0_rkind * (mmpp(ii) - MMl) * iZbs(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * &
                  iZij(1:Ntz, lvol)) / DDl &
                  + Zij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * sini(1:Ntz, ii)
            end if ! end of if( irz==0 )
          else ! take derivatives wrt Rbs and Zbc
            if (irz == 0) then
              dII(1:Ntz) = +im(ii) * cosi(1:Ntz, ii) * (XX(1:Ntz) - MMl * iRij(1:Ntz, lvol)) &
                  - 2.0_rkind * (mmpp(ii) - MMl) * iRbs(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * &
                  iZij(1:Ntz, lvol)) / DDl &
                  + Rij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * sini(1:Ntz, ii)
            else
              dII(1:Ntz) = -im(ii) * sini(1:Ntz, ii) * (YY(1:Ntz) - MMl * iZij(1:Ntz, lvol)) &
                  - 2.0_rkind * (mmpp(ii) - MMl) * iZbc(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * &
                  iZij(1:Ntz, lvol)) / DDl &
                  + Zij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * cosi(1:Ntz, ii)
            end if !matches ( irz==0 )
          end if! matches ( issym==0 )

        else

          dII(1:Ntz) = 0.0_rkind ! either no spectral constraint, or not the appropriate interface

        end if ! end of if( innout==1 .and. iocons==1 )

        constraint(1:Ntz) = + (dRij(1:Ntz, lvol) * tRij(1:Ntz, lvol - 1 + iocons) &
            + dZij(1:Ntz, lvol) * tZij(1:Ntz, lvol - 1 + iocons)) / length(1:Ntz)

        if (iocons == 0) then ! take derivatives of constraints at inner boundary

          if (innout == 0) then ! derivative wrt inner boundary coefficient
            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tRij(1:Ntz, lvol - 1) - dRij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) &
                    / length(1:Ntz) &
                    + constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tZij(1:Ntz, lvol - 1) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) &
                    / length(1:Ntz) &
                    + constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            else ! if issym==1 ; take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tRij(1:Ntz, lvol - 1) + dRij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) &
                    / length(1:Ntz) &
                    + constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tZij(1:Ntz, lvol - 1) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) &
                    / length(1:Ntz) &
                    + constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            end if
          else ! if innout==1 ; derivative wrt outer boundary coefficient
            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tRij(1:Ntz, lvol - 1)) / length(1:Ntz) &
                    - constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tZij(1:Ntz, lvol - 1)) / length(1:Ntz) &
                    - constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            else ! if issym==1 ; take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tRij(1:Ntz, lvol - 1)) / length(1:Ntz) &
                    - constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tZij(1:Ntz, lvol - 1)) / length(1:Ntz) &
                    - constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            end if
          end if

        else ! if iocons==1 ; take derivatives of constraints at outer boundary

          if (innout == 0) then ! derivative wrt inner boundary coefficient
            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tRij(1:Ntz, lvol)) / length(1:Ntz) &
                    + constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tZij(1:Ntz, lvol)) / length(1:Ntz) &
                    + constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            else ! take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tRij(1:Ntz, lvol)) / length(1:Ntz) &
                    + constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tZij(1:Ntz, lvol)) / length(1:Ntz) &
                    + constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            end if
          else ! if innout==1 ; derivative wrt outer boundary coefficient
            if (Igeometry == 3 .and. lvol == 1) then ! need to accomodate derivatives of coordinate axis

              if (issym == 0) then ! take derivatives wrt Rbc and Zbs
                if (irz == 0) then
                  dLL(1:Ntz) = ( & ! d/dRbc
                      + (cosi(1:Ntz, ii) - dRodR(1:Ntz, 0, ii)) * tRij(1:Ntz, lvol) - dRij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii) &
                      + (-dZodR(1:Ntz, 0, ii)) * tZij(1:Ntz, lvol) &
                      - constraint(1:Ntz) &
                      * (dRij(1:Ntz, lvol) * (cosi(1:Ntz, ii) - dRodR(1:Ntz, 0, ii)) &
                      + dZij(1:Ntz, lvol) * (-dZodR(1:Ntz, 0, ii)) ) / length(1:Ntz)) / length(1:Ntz)
                else
                  dLL(1:Ntz) = ( & ! d/dZbs
                      + (-dRodZ(1:Ntz, 1, ii)) * tRij(1:Ntz, lvol) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii) &
                      + (sini(1:Ntz, ii) - dZodZ(1:Ntz, 1, ii)) * tZij(1:Ntz, lvol) &
                      - constraint(1:Ntz) &
                      * (dRij(1:Ntz, lvol) * (-dRodZ(1:Ntz, 1, ii)) &
                      + dZij(1:Ntz, lvol) * (sini(1:Ntz, ii) - dZodZ(1:Ntz, 1, ii)) ) / length(1:Ntz)) / length(1:Ntz)
                end if ! end of if( irz==0 )
              else
                if (irz == 0) then
                  dLL(1:Ntz) = ( & ! d/dRbs
                      + (sini(1:Ntz, ii) - dRodR(1:Ntz, 1, ii)) * tRij(1:Ntz, lvol) + dRij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii) &
                      + (-dZodR(1:Ntz, 1, ii)) * tZij(1:Ntz, lvol) &
                      - constraint(1:Ntz) &
                      * (dRij(1:Ntz, lvol) * (sini(1:Ntz, ii) - dRodR(1:Ntz, 1, ii)) &
                      + dZij(1:Ntz, lvol) * (-dZodR(1:Ntz, 1, ii)) ) / length(1:Ntz)) / length(1:Ntz)
                else
                  dLL(1:Ntz) = ( & ! d/dZbs
                      + (-dRodZ(1:Ntz, 0, ii)) * tRij(1:Ntz, lvol) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii) &
                      + (cosi(1:Ntz, ii) - dZodZ(1:Ntz, 0, ii)) * tZij(1:Ntz, lvol) &
                      - constraint(1:Ntz) &
                      * (dRij(1:Ntz, lvol) * (-dRodZ(1:Ntz, 0, ii)) &
                      + dZij(1:Ntz, lvol) * (cosi(1:Ntz, ii) - dZodZ(1:Ntz, 0, ii)) ) / length(1:Ntz)) / length(1:Ntz)
                end if ! end of if( irz==0 )
              end if
            else
              if (issym == 0) then ! take derivatives wrt Rbc and Zbs
                if (irz == 0) then
                  dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tRij(1:Ntz, lvol) - dRij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) &
                      / length(1:Ntz) &
                      - constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
                else
                  dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tZij(1:Ntz, lvol) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) &
                      / length(1:Ntz) &
                      - constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
                end if
              else ! take derivatives wrt Rbs and Zbc
                if (irz == 0) then
                  dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tRij(1:Ntz, lvol) + dRij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) &
                      / length(1:Ntz) &
                      - constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
                else
                  dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tZij(1:Ntz, lvol) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) &
                      / length(1:Ntz) &
                      - constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
                end if ! end of if( irz==0 )
              end if ! end of if( issym==0 )
            end if ! end of if( Igeometry==3 .and. lvol==1 )
          end if ! end of if( innout==0 )

        end if ! end of if( iocons==0 )

      end if ! end of if( Igeometry.ge.3 )

      if (add_psifactor) then
        fact = psifactor(ii, lvol - 1 + innout)
      else
        fact = 1.0_rkind
      end if

      if (add_sweightfactor) then
        sfact = (lvol * 1.0_rkind / Nvol) ** wpoloidal ! * upsilon
      else
        sfact = 1.0_rkind
      end if

      ang_constr_jac(0, 1:Ntz, iocons, idof, innout, lvol) = -dII(1:Ntz) * fact
      ang_constr_jac(1, 1:Ntz, iocons, idof, innout, lvol) = -dLL(1:Ntz) * fact * sfact

    end do ! end of do iocons
  end subroutine angconst_evaluate_dBB

  subroutine alloc_for_hessian(ntz_loc)
    integer, intent(in) :: ntz_loc
    MPI_STATUS_TYPE :: stat

    call sallocate(dRodR, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
    call sallocate(dRodZ, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
    call sallocate(dZodR, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
    call sallocate(dZodZ, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)

    call sallocate(dRadR, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
    call sallocate(dRadZ, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
    call sallocate(dZadR, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
    call sallocate(dZadZ, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
  end subroutine alloc_for_hessian

  subroutine dealloc_hessian
    deallocate(force_ft_trunc_grad)
  end subroutine dealloc_hessian

  subroutine alloc_allrzrz(Nvol, MNtor, MMpol)
    integer :: MNMAX, MNtor, MMpol, Nvol

    MNMAX = MNtor + 1 + MMpol * (2 * MNtor + 1)

    if (allocated(mmRZRZ)) deallocate(mmRZRZ, nnRZRZ, allRZRZ)

    allocate(mmRZRZ(1:MNMAX), nnRZRZ(1:MNMAX), allRZRZ(1:4, 1:Nvol, 1:MNMAX))
    mmRZRZ(:) = 0
    nnRZRZ(:) = 0
    allRZRZ(:, :, :) = 0._rkind
  end subroutine alloc_allrzrz

  subroutine get_vec_pot(sumLrad, allAte, allAze, allAto, allAzo)
    integer :: vvol, ii, ideriv, sumLrad
    real(rkind), intent(out) :: allAte(1:sumLrad, 1:mn), allAze(1:sumLrad, 1:mn), allAto(1:sumLrad, 1:mn), allAzo(1:sumLrad, 1:mn)

    ideriv = 0

    sumLrad = 1
    do vvol = 1, Mvol
      do ii = 1, mn
        allAte(sumLrad:sumLrad + Lrad(vvol), ii) = Ate(vvol, ideriv, ii)%s(0:Lrad(vvol))
        allAze(sumLrad:sumLrad + Lrad(vvol), ii) = Aze(vvol, ideriv, ii)%s(0:Lrad(vvol))
        allAto(sumLrad:sumLrad + Lrad(vvol), ii) = Ato(vvol, ideriv, ii)%s(0:Lrad(vvol))
        allAzo(sumLrad:sumLrad + Lrad(vvol), ii) = Azo(vvol, ideriv, ii)%s(0:Lrad(vvol))
      end do
      sumLrad = sumLrad + Lrad(vvol) + 1
    end do
  end subroutine get_vec_pot

  subroutine get_grid_data(sumLrad, Rij_grid, Zij_grid, sg_grid, BR_grid, Bp_grid, BZ_grid)
    integer :: sumLrad, alongLrad, Ngrid_local, Ngrid_sum
    integer :: vvol, ii, jj, kk, jk, Lcurvature
    real(rkind) :: lss, teta, zeta, st(1:Node), Bst(1:Node), gBzeta

    real(rkind), intent(out) :: Rij_grid(1:sumLrad, 1:Ntz), Zij_grid(1:sumLrad, 1:Ntz), sg_grid(1:sumLrad, 1:Ntz), &
        & BR_grid(1:sumLrad, 1:Ntz), Bp_grid(1:sumLrad, 1:Ntz), BZ_grid(1:sumLrad, 1:Ntz)

    if (myid == 0) then

      ijreal(1:Ntz) = 0.0_rkind
      ijimag(1:Ntz) = 0.0_rkind
      jireal(1:Ntz) = 0.0_rkind

      Ngrid_sum = 0

      do vvol = 1, Mvol

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! sets Lcoordinatesingularity and Lplasmaregion

        if (Ngrid < 0) then
          Ngrid_local = Lrad(vvol)! default
        else
          Ngrid_local = Ngrid
        end if
        if (Ngrid_local == 0) cycle ! nothing to output

        do ii = 0, Ngrid_local ! sub-grid
          lss = ii * 2.0_rkind / Ngrid_local - 1.0_rkind
          if (Lcoordinatesingularity .and. ii == 0) then
            Lcurvature = 0 ! Jacobian is not defined
          else
            Lcurvature = 1 ! compute Jacobian
          end if

          call compute_coordinates(vvol, lss, Lcurvature, Nt, Nz, mn)! only Rij(0,:) and Zij(0,:) are required; Rmn & Zmn are available

          alongLrad = Ngrid_sum + ii + 1

          Rij_grid(alongLrad, 1:Ntz) = Rij(1:Ntz, 0, 0)
          Zij_grid(alongLrad, 1:Ntz) = Zij(1:Ntz, 0, 0)
          sg_grid(alongLrad, 1:Ntz) = sg(1:Ntz, 0)

          if (Lcurvature == 1) then

            select case (Igeometry)

            case (3)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = (/lss, teta/)
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta * inv_jac(jk)! BR
                  ijimag(jk) = (1.0_rkind) * gBzeta * inv_jac(jk)! Bp
                  jireal(jk) = (Zij(jk, 1, 0) * Bst(1) + Zij(jk, 2, 0) * Bst(2) + Zij(jk, 3, 0) * 1.0_rkind) * gBzeta * inv_jac(jk)! BZ
                end do
              end do

            case (1)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = (/lss, teta/)
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta * inv_jac(jk)! BR
                  ijimag(jk) = (rpol) * gBzeta * inv_jac(jk)! Bzeta
                  jireal(jk) = (+rtor * Bst(2)) * gBzeta * inv_jac(jk)! Btheta
                end do
              end do

            case (2)
              do kk = 0, Nz - 1
                zeta = kk * pi2nfp / Nz
                do jj = 0, Nt - 1
                  teta = jj * pi2 / Nt
                  jk = 1 + jj + kk * Nt
                  st(1:2) = (/lss, teta/)
                  call compute_magnetic_field(vvol, zeta, st(1:Node), Bst(1:Node), gBzeta)
                  ijreal(jk) = (Rij(jk, 1, 0) * Bst(1) + Rij(jk, 2, 0) * Bst(2) + Rij(jk, 3, 0) * 1.0_rkind) * gBzeta * inv_jac(jk)! BR
                  ijimag(jk) = (1.0_rkind) * gBzeta * inv_jac(jk)! Bp
                  jireal(jk) = (Bst(2)) * gBzeta * inv_jac(jk)! BZ
                end do
              end do

            end select !Igeometry
          end if ! end of if( Lcurvature==1 )

          BR_grid(alongLrad, 1:Ntz) = ijreal(1:Ntz)
          Bp_grid(alongLrad, 1:Ntz) = ijimag(1:Ntz)
          BZ_grid(alongLrad, 1:Ntz) = jireal(1:Ntz)

        end do ! end of do ii

        Ngrid_sum = Ngrid_sum + Ngrid_local + 1 ! offset for storing data

      end do ! end of do vvol

    end if
  end subroutine get_grid_data

  subroutine alloc_remn_ext
    if (.not.allocated(Remn_ext)) allocate(Remn_ext(1:mn, 1:Iquad(1), 0:2))
    if (.not.allocated(Romn_ext)) allocate(Romn_ext(1:mn, 1:Iquad(1), 0:2))
    if (.not.allocated(Zemn_ext)) allocate(Zemn_ext(1:mn, 1:Iquad(1), 0:2))
    if (.not.allocated(Zomn_ext)) allocate(Zomn_ext(1:mn, 1:Iquad(1), 0:2))
  end subroutine alloc_remn_ext

  subroutine get_mesh(vvol, Rij_grid, Zij_grid, ind_i, ind_j)
    integer :: vvol, ii, ind_i, ind_j, lcurv
    real(rkind) :: lss

    real(rkind), intent(out) :: Rij_grid(1:Iquad(vvol), 1:Ntz), Zij_grid(1:Iquad(vvol), 1:Ntz)

    call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

    if (myid == 0) then

      do ii = 1, Iquad(vvol)

        lss = gaussianabscissae(ii, vvol)

        call compute_coordinates(vvol, lss, 2, Nt, Nz, mn)! only Rij(0,:) and Zij(0,:) are required; Rmn & Zmn are available
        Rij_grid(ii, 1:Ntz) = Rij(1:Ntz, ind_i, ind_j)
        Zij_grid(ii, 1:Ntz) = Zij(1:Ntz, ind_i, ind_j)
      end do ! end of do ii
    end if
  end subroutine get_mesh

  subroutine update_bnorm
    integer :: vvol

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    if (Lbnszero) then
      iBns(1:mn) = 0.0_rkind
      if (.not.enforce_stell_sym) iBnc(1:mn) = 0.0_rkind
    else
      select case ( Igeometry ) ! 08 Feb 16
      case ( 1 )
        tflux(1) = dtflux(1)
        pflux(1) = dpflux(1)! 08 Feb 16
      case ( 2:3 )
        tflux(1) = dtflux(1)
        pflux(1) = 0.0_rkind ! 08 Feb 16
      end select ! 08 Feb 16

      do vvol = 2, Mvol
        tflux(vvol) = tflux(vvol - 1) + dtflux(vvol)! 01 Jul 14
        pflux(vvol) = pflux(vvol - 1) + dpflux(vvol)! 01 Jul 14
      end do

      tflux(1:Mvol) = tflux(1:Mvol) * pi2 / phiedge ! this is the "inverse" operation defined in preset; 19 Jul 16
      pflux(1:Mvol) = pflux(1:Mvol) * pi2 / phiedge

      call compute_bnorm_plasmab(mn, Ntz, efmn(1:mn), ofmn(1:mn))! compute normal field etc. on computational boundary

      iBns(2:mn) = gBnbld * iBns(2:mn) + ( 1.0_rkind - gBnbld ) * ofmn(2:mn)
      if (.not.enforce_stell_sym) iBnc(1:mn) = gBnbld * iBnc(1:mn) + ( 1.0_rkind - gBnbld ) * efmn(1:mn)

      if (mn == 1) then
        if ( enforce_stell_sym ) then
          bnserr = 0.0 !TODO: NOT SURE, this should test if bns is actually 0
        else
          bnserr = sum(abs(iBnc(1:mn) - efmn(1:mn))) / (mn )
        end if
      else
        if ( enforce_stell_sym ) then
          bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1)
        else
          bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1) + sum(abs(iBnc(1:mn) - efmn(1:mn))) / (mn )
        end if
      end if

    end if

  end subroutine update_bnorm

  subroutine calc_beltrami_error(lvol, error_type, lquad, jerroravg, jerrormax, err_grid)
    integer, intent(in) :: lvol, error_type, lquad
    real(rkind), intent(out) :: jerroravg(1:3), jerrormax(1:3), err_grid(1:lquad + 1, 1:3, 1:Ntz)

    integer :: jquad, ll, ii, jj, kk, uu, ideriv, twolquad, mm

    real(rkind) :: lss, sbar, gBu(1:Ntz, 1:3, 0:3), gJu(1:Ntz, 1:3), intvol
    real(rkind) :: B_cartesian(1:Ntz, 1:3), J_cartesian(1:Ntz, 1:3)

    real(rkind) :: Atemn(1:mn, 0:2), Azemn(1:mn, 0:2), Atomn(1:mn, 0:2), Azomn(1:mn, 0:2)

    integer :: itype, icdgqf
    real(rkind) :: aa, bb, cc, dd, weight(1:lquad + 1), abscis(1:lquad), workfield(1:2 * lquad)
    

    call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

    itype = 1
    aa = -1.0_rkind
    bb = +1.0_rkind
    cc = 0.0_rkind
    dd = 0.0_rkind
    twolquad = 2 * lquad

    call CDGQF(lquad, abscis(1:lquad), weight(1:lquad), itype, aa, bb, twolquad, workfield(1:twolquad), icdgqf)! prepare Gaussian quadrature
    weight(lquad + 1) = 0.0_rkind

    call fatal(icdgqf /= 0, __FILE__, "failed to construct Gaussian integration abscisae and weights")

    jerroravg(1:3) = 0.0_rkind
    ideriv = 0 ! three components of the error in \curl B - mu B; initialize summation
    jerrormax(1:3) = 0.0_rkind
    intvol = 0.0_rkind

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    do jquad = 1, lquad + 1 ! loop over radial sub-sub-grid (numerical quadrature)

      if (jquad == lquad + 1) then
        lss = 1.0_rkind
        sbar = 1.0_rkind
      else
        lss = abscis(jquad)
        sbar = ( lss + 1.0_rkind ) * 0.5_rkind
      end if

      call compute_coordinates(lvol, lss, 2, Nt, Nz, mn)! returns coordinates, metrics, . . .

      if (Lcoordinatesingularity) then ! Zernike 1 Jul 2019
        call get_zernike_d2(sbar, Lrad(lvol), mpol, zernike)
      else
        call get_cheby_d2(lss, Lrad(lvol), cheby(0:Lrad(lvol), 0:2))
      end if

      Atemn(1:mn, 0:2) = 0.0_rkind ! initialize summation over Chebyshev/Zernike polynomials
      Azemn(1:mn, 0:2) = 0.0_rkind
      if (.not.enforce_stell_sym ) then
        Atomn(1:mn, 0:2) = 0.0_rkind
        Azomn(1:mn, 0:2) = 0.0_rkind
      else
        Atomn(1:mn, 0:2) = 0.0_rkind ! these are used below
        Azomn(1:mn, 0:2) = 0.0_rkind
      end if

      if (Lcoordinatesingularity) then
        do ll = 0, Lrad(lvol)! radial (Chebyshev) resolution of magnetic vector potential

          do ii = 1, mn ! Fourier resolution of magnetic vector potential
            mm = im(ii)
            if (ll < mm) cycle
            if (mod(ll + mm, 2) > 0) cycle

            Atemn(ii, 0) = Atemn(ii, 0) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
            Atemn(ii, 1) = Atemn(ii, 1) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 0.5_rkind
            Atemn(ii, 2) = Atemn(ii, 2) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 0.5_rkind * 0.5_rkind

            Azemn(ii, 0) = Azemn(ii, 0) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
            Azemn(ii, 1) = Azemn(ii, 1) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 0.5_rkind
            Azemn(ii, 2) = Azemn(ii, 2) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 0.5_rkind * 0.5_rkind

            if (.not.enforce_stell_sym ) then

              Atomn(ii, 0) = Atomn(ii, 0) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
              Atomn(ii, 1) = Atomn(ii, 1) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 0.5_rkind
              Atomn(ii, 2) = Atomn(ii, 2) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 0.5_rkind * 0.5_rkind

              Azomn(ii, 0) = Azomn(ii, 0) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
              Azomn(ii, 1) = Azomn(ii, 1) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 0.5_rkind
              Azomn(ii, 2) = Azomn(ii, 2) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 0.5_rkind * 0.5_rkind

            end if ! end of if( .not. enforce_stell_sym) ; 20 Jun 14

          end do ! end of do ii

        end do ! end of do ll

      else

        do ll = 0, Lrad(lvol)! radial (Chebyshev) resolution of magnetic vector potential

          do ii = 1, mn ! Fourier resolution of magnetic vector potential

            Atemn(ii, 0) = Atemn(ii, 0) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
            Atemn(ii, 1) = Atemn(ii, 1) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
            Atemn(ii, 2) = Atemn(ii, 2) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

            Azemn(ii, 0) = Azemn(ii, 0) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
            Azemn(ii, 1) = Azemn(ii, 1) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
            Azemn(ii, 2) = Azemn(ii, 2) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

            if (.not.enforce_stell_sym ) then

              Atomn(ii, 0) = Atomn(ii, 0) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
              Atomn(ii, 1) = Atomn(ii, 1) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
              Atomn(ii, 2) = Atomn(ii, 2) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

              Azomn(ii, 0) = Azomn(ii, 0) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
              Azomn(ii, 1) = Azomn(ii, 1) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
              Azomn(ii, 2) = Azomn(ii, 2) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

            end if ! end of if( .not. enforce_stell_sym) ; 20 Jun 14

          end do ! end of do ii

        end do ! end of do ll
      end if

      ofmn(1:mn) = -im(1:mn) * Azemn(1:mn, 0) - in(1:mn) * Atemn(1:mn, 0)
      efmn(1:mn) = +im(1:mn) * Azomn(1:mn, 0) + in(1:mn) * Atomn(1:mn, 0)
      sfmn(1:mn) = -im(1:mn) * Azemn(1:mn, 1) - in(1:mn) * Atemn(1:mn, 1)
      cfmn(1:mn) = +im(1:mn) * Azomn(1:mn, 1) + in(1:mn) * Atomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 1, 0), gBu(1:Ntz, 1, 1&
          ))!  (gB^s)   , d(gB^s)/ds

      efmn(1:mn) = -im(1:mn) * im(1:mn) * Azemn(1:mn, 0) - im(1:mn) * in(1:mn) * Atemn(1:mn, 0)
      ofmn(1:mn) = -im(1:mn) * im(1:mn) * Azomn(1:mn, 0) - im(1:mn) * in(1:mn) * Atomn(1:mn, 0)
      cfmn(1:mn) = +in(1:mn) * im(1:mn) * Azemn(1:mn, 0) + in(1:mn) * in(1:mn) * Atemn(1:mn, 0)
      sfmn(1:mn) = +in(1:mn) * im(1:mn) * Azomn(1:mn, 0) + in(1:mn) * in(1:mn) * Atomn(1:mn, 0)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 1, 2), gBu(1:Ntz, 1, 3&
          ))! d(gB^s)/dt, d(gB^s)/dz

      efmn(1:mn) = -Azemn(1:mn, 1)
      ofmn(1:mn) = -Azomn(1:mn, 1)
      cfmn(1:mn) = -Azemn(1:mn, 2)
      sfmn(1:mn) = -Azomn(1:mn, 2)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 2, 0), gBu(1:Ntz, 2, 1&
          ))!  (gB^t)   , d(gB^t)/ds

      ofmn(1:mn) = +im(1:mn) * Azemn(1:mn, 1)
      efmn(1:mn) = -im(1:mn) * Azomn(1:mn, 1)
      sfmn(1:mn) = -in(1:mn) * Azemn(1:mn, 1)
      cfmn(1:mn) = +in(1:mn) * Azomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 2, 2), gBu(1:Ntz, 2, 3&
          ))! d(gB^t)/dt, d(gB^t)/dz

      efmn(1:mn) = +Atemn(1:mn, 1)
      ofmn(1:mn) = +Atomn(1:mn, 1)
      cfmn(1:mn) = +Atemn(1:mn, 2)
      sfmn(1:mn) = +Atomn(1:mn, 2)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 3, 0), gBu(1:Ntz, 3, 1&
          ))!  (gB^z)   , d(gB^z)/ds

      ofmn(1:mn) = -im(1:mn) * Atemn(1:mn, 1)
      efmn(1:mn) = +im(1:mn) * Atomn(1:mn, 1)
      sfmn(1:mn) = +in(1:mn) * Atemn(1:mn, 1)
      cfmn(1:mn) = -in(1:mn) * Atomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 3, 2), gBu(1:Ntz, 3, 3&
          ))! d(gB^z)/dt, d(gB^z)/dz

      do ii = 1, 3

        select case ( ii )
        case ( 1 )
          jj = 2
          kk = 3
        case ( 2 )
          jj = 3
          kk = 1
        case ( 3 )
          jj = 1
          kk = 2
        end select

        gJu(1:Ntz, ii) = 0.0_rkind

        do uu = 1, 3 ! summation over uu

          gJu(1:Ntz, ii) = gJu(1:Ntz, ii) &
              + ( gBu(1:Ntz, uu, jj) * guvij(1:Ntz, uu, kk, 0) &
              + gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, kk, jj) &
              - gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, kk, 0) * sg(1:Ntz, jj) * inv_jac(1:Ntz) &
              ) &
              - ( gBu(1:Ntz, uu, kk) * guvij(1:Ntz, uu, jj, 0) &
              + gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, jj, kk) &
              - gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, jj, 0) * sg(1:Ntz, kk) * inv_jac(1:Ntz) &
              )
        end do ! end of do uu

        gJu(1:Ntz, ii) = gJu(1:Ntz, ii) * inv_jac(1:Ntz)

      end do ! end of do ii

      B_cartesian(:, 1) = (gBu(1:Ntz, 1, 0) * Rij(1:Ntz, 1, 0) + gBu(1:Ntz, 2, 0) * Rij(1:Ntz, 2, 0) &
          + gBu(1:Ntz, 3, 0) * Rij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
      B_cartesian(:, 2) = (gBu(1:Ntz, 1, 0) * Zij(1:Ntz, 1, 0) + gBu(1:Ntz, 2, 0) * Zij(1:Ntz, 2, 0) &
          + gBu(1:Ntz, 3, 0) * Zij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
      B_cartesian(:, 3) = ( gBu(1:Ntz, 3, 0) * Rij(1:Ntz, 0, 0)) * inv_jac(1:Ntz)

      J_cartesian(:, 1) = (gJu(1:Ntz, 1) * Rij(1:Ntz, 1, 0) + gJu(1:Ntz, 2) * Rij(1:Ntz, 2, 0) + gJu(1:Ntz, 3) * Rij(1:Ntz, 3, 0)&
          &) * inv_jac(1:Ntz)
      J_cartesian(:, 2) = (gJu(1:Ntz, 1) * Zij(1:Ntz, 1, 0) + gJu(1:Ntz, 2) * Zij(1:Ntz, 2, 0) + gJu(1:Ntz, 3) * Zij(1:Ntz, 3, 0)&
          &) * inv_jac(1:Ntz)
      J_cartesian(:, 3) = ( gJu(1:Ntz, 3) * Rij(1:Ntz, 0, 0)) * inv_jac(1:Ntz)

      if (error_type == 1 .and. Igeometry == 3) then
        do ii = 1, 3
          err_grid(jquad, ii, 1:Ntz) = sg(1:Ntz, 0) * abs(J_cartesian(1:Ntz, ii) - mu(lvol) * B_cartesian(1:Ntz, ii))
          jerroravg(ii) = jerroravg(ii) &
              + weight(jquad) * sum(sg(1:Ntz, 0) * abs(J_cartesian(1:Ntz, ii) - mu(lvol) * B_cartesian(1:Ntz, ii)))
          jerrormax(ii) = max(jerrormax(ii), maxval(abs(J_cartesian(1:Ntz, ii) - mu(lvol) * B_cartesian(1:Ntz, ii))))
        end do
      else
        do ii = 1, 3
          err_grid(jquad, ii, 1:Ntz) = abs(gJu(1:Ntz, ii) - mu(lvol) * gBu(1:Ntz, ii, 0))
          jerroravg(ii) = jerroravg(ii) + weight(jquad) * sum(abs(gJu(1:Ntz, ii) - mu(lvol) * gBu(1:Ntz, ii, 0)))
          jerrormax(ii) = max(jerrormax(ii), maxval(abs(gJu(1:Ntz, ii) - mu(lvol) * gBu(1:Ntz, ii, 0)) * inv_jac(1:Ntz)))
        end do
      end if
      intvol = intvol + weight(jquad) * sum(sg(1:Ntz, 0))

    end do ! end of do jquad

    jerroravg(1:3) = jerroravg(1:3) / intvol

    ! if (error_type == 1 .and. Igeometry == 3) then
    !   write(stdout, 1002)lvol, Lrad(lvol), jerroravg(1:3)! write error to screen
    !   write(stdout, 1003)lvol, Lrad(lvol), jerrormax(1:3)! write error to screen
    ! else
    !   write(stdout, 1004)lvol, Lrad(lvol), jerroravg(1:3)! write error to screen
    !   write(stdout, 1005)lvol, Lrad(lvol), jerrormax(1:3)! write error to screen
    ! end if
    ! 1002 format("jo00aa : ; lvol =", i3, " ; lrad =", i3, " ; AVG E^\R="es23.15" , E^\Z="es23.15" , E^\phi="es23.15)
    ! 1003 format("jo00aa : ; lvol =", i3, " ; lrad =", i3, " ; MAX E^\R="es23.15" , E^\Z="es23.15" , E^\phi="es23.15)
    ! 1004 format("jo00aa : ; lvol =", i3, " ; lrad =", i3, " ; AVG E^\s="es23.15" , E^\t="es23.15" , E^\z="es23.15)
    ! 1005 format("jo00aa : ; lvol =", i3, " ; lrad =", i3, " ; MAX E^\s="es23.15" , E^\t="es23.15" , E^\z="es23.15)

  end subroutine calc_beltrami_error

  subroutine print_threads

    !$OMP PARALLEL
    write(*,*)"OMP process ", OMP_GET_THREAD_NUM(), " / ", OMP_GET_NUM_THREADS()
    !$OMP END PARALLEL
  end subroutine print_threads

  subroutine prepare_endfile_writing

    integer :: ii, mm, nn

    Isurf(1:Mvol) = IPDt(1:Mvol)

    if (Lfreebound == 1 .or. Zbs(0, 1) > 0.0_rkind) then
      do ii = 1, mn
        mm = im(ii)
        nn = in(ii) / Nfp
        Rbc(nn, mm) = iRbc(ii, Nvol)
        Zbs(nn, mm) = iZbs(ii, Nvol)
        Rbs(nn, mm) = iRbs(ii, Nvol)
        Zbc(nn, mm) = iZbc(ii, Nvol)

        Bns(nn, mm) = iBns(ii)
        Bnc(nn, mm) = iBnc(ii)
      end do
    end if

    Rac(0:Ntor) = iRbc(1:Ntor + 1, 0)
    Zas(0:Ntor) = iZbs(1:Ntor + 1, 0)
    Ras(0:Ntor) = iRbs(1:Ntor + 1, 0)
    Zac(0:Ntor) = iZbc(1:Ntor + 1, 0)

  end subroutine prepare_endfile_writing

end module wrapper_funcs_mod
