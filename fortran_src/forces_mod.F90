module forces_mod
  ! allow(C121)
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind, logtolerance, small
  use io_mod, only: ounit, munit, stderr
  use input_list_mod, only: Wmacros, Wdforce, Wdfp200, Nvol, Ntor, Lrad, Igeometry, epsilon, Lconstraint, Lcheck, mupftol, &
      enforce_stell_sym, Lfreebound, LHmatrix, gamma, pscale, adiabatic, Mpol, tflux, mu, &
      Wdfp100, Isurf, curpol, Wlforce, gBnbld, gBntol
  use allglobal_mod, only: ncpu, myid, cpus, Mvol, NAdof, iRbc, iZbs, iRbs, iZbc, ImagneticOK, Energy, ForceErr, &
      Lcoordinatesingularity, Lplasmaregion, Lvacuumregion, mn, dpflux, dtflux, sweight, Bemn, Bomn, Iomn, &
      Iemn, Somn, Semn, BBe, IIo, BBo, IIe, LGdof, dBdX, Ate, Aze, Ato, Azo, lBBintegral, dFFdRZ, HdFFdRZ, &
      dBBdmp, dmupfdx, force_ft_trunc_grad, force_ft_trunc_grad_alloc, hessian2D, Lhessian2Dallocated, &
      denergydrr, denergydzr, BBweight, LocalConstraint, xoffset, solution, IPdtdPf, IsMyVolume, im, in, &
      IsMyVolumeValue, WhichCpuID, ext, MPI_COMM_SPEC, Nt, Nz, Ntz, Iquad, Localconstraint, IPDt, IPDtdPf, Nt, Nz, &
      Lsavedguvij, total_pflux, dMA, dMD, diotadxup, dItGpdxtp, psifactor, lmns, ijreal, efmn, ofmn, cfmn, &
      sfmn, evmn, odmn, comn, simn, cosi, sini, mmpp, vvolume, dvolume, Rij, Zij, iRij, iZij, dRij, dZij, &
      tRij, tZij, dFFdRZ, dRodR, dRodZ, dZodR, dZodZ, ijimag, sg, inv_jac, guvij, Pomn, Pemn, build_vector_potential, &
      bnserr, iBns, iBnc, update_bnorm_inbeltrami, pi2nfp, gvuij
  use memory_allocation_mod, only: sallocate, nallocate
  use errors_mod, only: fatal
  use mpi_collectives_mod, only: broadcast
  use volume_mod, only: lregion, volume
  use profiles_distribution_mod, only: broadcast_profile_data
  use constants_mod, only: pi2
  use plasma_current_mod, only: compute_plasma_current, lbpol
  use typedefns_mod, only: MatrixLU
  use numerical_mod, only: tfft, invfft, set_fftw_plan
  use intghs_mod, only: intghs, intghs_workspace_init, intghs_workspace_destroy
  use geometry_mod, only: compute_coordinates
  use chebyshev_mod, only: volume_integrate_chebyshev
  use beltrami_field_mod, only: construct_beltrami_field
  use memory_mod, only: allocate_Beltrami_matrices, allocate_geometry_matrices, deallocate_Beltrami_matrices, &
      deallocate_geometry_matrices
  use metric_mod, only: compute_guvijsave
  use matrices_mod, only: matrix, mtrxhs
  use packing_mod, only: packab, packxi
  use magnetic_field_mod, only: compute_rotational_transform
  use freeb_mod, only: compute_bnorm_plasmab

  implicit none(type, external)

  private
  public :: dforce, dfp100, lforce, get_perturbed_solution, evaluate_dmupfdx, get_LU_beltrami_matrices

  external :: DGESV, DGETRS, daxpy, DGETRF, DCOPY

contains

  subroutine dforce(NGdof, position, force_ft_trunc, LComputeDerivatives, LComputeAxis)
    integer, intent(in) :: NGdof ! dimensions
    real(rkind), intent(in) :: position(0:NGdof)
    real(rkind), intent(out) :: force_ft_trunc(0:NGdof)
    logical, intent(in) :: LComputeDerivatives !
    logical, intent(inout) :: LComputeAxis

    real(rkind) :: cput
    integer :: vvol, ii, irz, issym, tdoc, idoc, idof, tdof, ivol, Ndofgl
    integer :: NN, cpu_id, ideriv
    real(rkind) :: Xdof(1:Mvol - 1)
    integer :: ipiv(1:Mvol)
    real(rkind), allocatable :: Fvec(:), dpfluxout(:)

    integer :: idgesv
    integer :: idofr, idofz, tdofr, tdofz

    character :: packorunpack

    logical :: dfp100_logical

    if (LocalConstraint) then
      call sallocate(dmupfdx, 1, Mvol, 1, 1, 1, 2, 1, LGdof, 0, 1, 0.0_rkind)
    else
      call sallocate(dmupfdx, 1, Mvol, 1, Mvol - 1, 1, 2, 1, LGdof, 0, 1, 0.0_rkind)
    end if

    force_ft_trunc_grad_alloc = .false.

    if ( LcomputeDerivatives ) then
      call sallocate(dFFdRZ, 1, LGdof, 0, 1, 1, LGdof, 0, 1, 1, Mvol, 0.0_rkind)
      call sallocate(dBBdmp, 1, LGdof, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)
      call sallocate(force_ft_trunc_grad, 1, NGdof, 1, NGdof, 0.0_rkind)
      force_ft_trunc_grad_alloc = .true.
    end if

    ! Unpack position to generate arrays iRbc, iZbs, IRbs, iZbc.
    ! This also computes the axis geometry.
    packorunpack = 'U' ! unpack geometrical degrees-of-freedom

    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol), iZbc(1:mn&
        & , 0:Mvol), packorunpack, LcomputeDerivatives, LComputeAxis)

    if (LcomputeDerivatives) then
      dBBdmp(1:LGdof, 1:Mvol, 0:1, 1:2) = 0.0_rkind
    end if

    ! SOLVE FIELD IN AGREEMENT WITH CONSTRAINTS AND GEOMETRY
    ! ------------------------------------------------------
    ! Two different cases, both with their own parallelization.
    ! If LOCAL constraint, each process iterates on the local poloidal flux and Lagrange multiplier
    ! to match the local constraint. Then all information is broadcasted by the master task
    ! If GLOBAL constraint, only the master task iterates on all Lagrange multipliers and poloidal
    ! fluxes. Over tasks are stuck in an infinite loop where they "help" the master task compute
    ! each iteration. At the last iteration, master task send IconstraintOK=.TRUE. to all tasks, and
    ! they exit their infinite loops.

    ! Local constraint case - simply call dfp100 and then dfp200
    Xdof(1:Mvol - 1) = 0.0_rkind
    if (LocalConstraint) then

      call sallocate(Fvec, 1, Mvol - 1, 0.0_rkind)

      Ndofgl = 0
      Fvec(1:Mvol - 1) = 0
      dfp100_logical = .false.
      Xdof(1:Mvol - 1) = dpflux(2:Mvol) + xoffset

      ! Solve for field
      dBdX%L = LComputeDerivatives
      call dfp100(Ndofgl, Xdof, Fvec, dfp100_logical)

      deallocate(Fvec)
      ! --------------------------------------------------------------------------------------------------
      ! Global constraint - call the master task calls hybrd1 on dfp100, others call dfp100_loop.
    else

      IPDtdPf = 0.0_rkind
      Xdof(1:Mvol - 1) = dpflux(2:Mvol) + xoffset

      if (Lfreebound == 1) then
        ! Mvol-1 surface current plus 1 poloidal linking current constraints
        Ndofgl = Mvol
      else
        ! add an additional constraint to make the total pflux = 0
        if (Igeometry == 1) then
          Ndofgl = Mvol
        else
          ! Mvol-1 surface current constraints
          Ndofgl = Mvol - 1
        end if
      end if

      call sallocate(Fvec, 1, Ndofgl, 0.0_rkind)

      dfp100_logical = .false.

      call dfp100(Ndofgl, Xdof(1:Mvol - 1), Fvec(1:Ndofgl), dfp100_logical)

      call sallocate(dpfluxout, 1, Ndofgl, 0.0_rkind)
      if (myid == 0) then

        dpfluxout = Fvec
        call DGESV(Ndofgl, 1, IPdtdPf, Ndofgl, ipiv, dpfluxout, Ndofgl, idgesv)

        ! First step Newton's method
        dpflux(2:Mvol) = dpflux(2:Mvol) - dpfluxout(1:Mvol - 1)

        if (Igeometry == 1) then
          dpflux(1) = dpflux(1) - dpfluxout(Mvol)
        end if

        if (Lfreebound == 1) then
          dtflux(Mvol) = dtflux(Mvol) - dpfluxout(Mvol)
        end if
      end if

      ! Broadcast the field and pflux
      call broadcast(dpfluxout(1:Ndofgl), Ndofgl, 0)
      call broadcast(dpflux(1:Mvol), Mvol, 0)
      if (Lfreebound == 1) then
        call broadcast(dtflux(Mvol), 0)
      end if

      do vvol = 2, Mvol

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")

        end if

        NN = NAdof(vvol)

        call sallocate(solution, 1, NN, 0, 2, 0.0_rkind)

        ! Pack field and its derivatives
        packorunpack = 'P'
        call packab(packorunpack, vvol, NN, solution(1:NN, 0), 0)! packing
        call packab(packorunpack, vvol, NN, solution(1:NN, 2), 2)! packing

        ! compute the field with renewed dpflux via single Newton method step
        if (Lfreebound == 1 .and. (vvol == Mvol) ) then
          call packab(packorunpack, vvol, NN, solution(1:NN, 1), 1)! packing
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(vvol - 1) * solution(1:NN, 2) & ! derivative w.r.t pflux
              - dpfluxout(vvol) * solution(1:NN, 1)! derivative w.r.t tflux
        else
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(vvol - 1) * solution(1:NN, 2)
        end if

        ! Unpack field in vector potential Fourier harmonics
        packorunpack = 'U'
        call packab(packorunpack, vvol, NN, solution(1:NN, 0), 0)! unpacking

        deallocate(solution)

      end do

      !add an additional constraint to make the total pflux 0
      if (Igeometry == 1) then

        vvol = 1

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then

        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")

        else
          NN = NAdof(vvol)

          call sallocate(solution, 1, NN, 0, 2, 0.0_rkind)

          ! Pack field and its derivatives
          packorunpack = 'P'
          call packab(packorunpack, vvol, NN, solution(1:NN, 0), 0)! packing
          call packab(packorunpack, vvol, NN, solution(1:NN, 2), 2)! packing

          ! compute the field with renewed dpflux via single Newton method step
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(Mvol) * solution(1:NN, 2)

          ! Unpack field in vector potential Fourier harmonics
          packorunpack = 'U'
          call packab(packorunpack, vvol, NN, solution(1:NN, 0), 0)! unpacking

          deallocate(solution)
        end if
      end if

      deallocate(Fvec)
      deallocate(dpfluxout)
    end if !matches if( LocalConstraint )

    ! --------------------------------------------------------------------------------------------------
    !                                    MPI COMMUNICATIONS

    ! Finally broadcast the field information to all tasks from the task which did the computation
    ! TODO: improve MPI communication
    do vvol = 1, Mvol
      cpu_id = WhichCpuID(vvol, ncpu)

      ! Broadcast all ImagneticOK
      call broadcast(ImagneticOK(vvol), cpu_id)

      do ideriv = 0, 2
        if ( (.not.LcomputeDerivatives) .and. (ideriv /= 0) ) cycle
        do ii = 1, mn
          call broadcast(Ate(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          call broadcast(Aze(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
        end do
      end do

      if (.not.enforce_stell_sym) then
        do ideriv = 0, 2
          if ( (.not.LcomputeDerivatives) .and. (ideriv /= 0) ) cycle
          do ii = 1, mn
            call broadcast(Ato(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
            call broadcast(Azo(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          end do
        end do
      end if
    end do

    ! Compute local force and derivatives
    !Compuatation of Hessian2d is cleared at this point.
    call dfp200(LcomputeDerivatives)

    ! Broadcast information to all CPUs
    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      call broadcast_profile_data(vvol)

    end do

    lBBintegral(1:Nvol) = lBBintegral(1:Nvol) * 1.0_rkind / 2.0_rkind
    Energy = sum(lBBintegral(1:Nvol))! should also compute beta

    ! ---------------
    ! CONSTRUCT FORCE
    ! ---------------

    force_ft_trunc(0:NGdof) = 0.0_rkind

    do vvol = 1, Mvol - 1

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      tdoc = (vvol - 1) * LGdof

      if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this
        ! interface are valid

        idoc = 0 ! degree-of-constraint counter: set

        force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn) = (Bemn(1:mn, vvol + 1, 0) - Bemn(1:mn, vvol + 0, 1)) * BBweight(1:mn)!
        !pressure imbalance

        BBe(vvol) = max(sum(abs(force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn))) / (mn) , logtolerance)! screen diagnostics

        idoc = idoc + mn ! degree-of-constraint counter: increment

        if (Igeometry >= 3) then ! add spectral constraints

          force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn - 1) = (Iomn(2:mn, vvol + 0)) * epsilon & ! spectral constraints
              + (+Somn(2:mn, vvol + 0, 1)) * sweight(vvol + 0) & ! poloidal length constraint
              - (Somn(2:mn, vvol + 1, 0)) * sweight(vvol + 1)

          IIo(vvol) = max(sum(abs(force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn - 1))) / (mn - 1) , logtolerance)! screen
          ! diagnostics

          idoc = idoc + mn - 1

        end if

        if (.not.enforce_stell_sym) then

          force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn - 1) = (Bomn(2:mn, vvol + 1, 0) - Bomn(2:mn, vvol + 0, 1)) *&
              & BBweight(2:mn)! pressure imbalance

          BBo(vvol) = max(sum(abs(force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn - 1))) / (mn - 1) , logtolerance)! screen
          ! diagnostics

          idoc = idoc + mn - 1 ! degree-of-constraint counter: increment

          if (Igeometry >= 3) then ! add spectral constraints

            force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn) = (Iemn(1:mn, vvol + 0)) * epsilon & ! spectral constraints
                + (+Semn(1:mn, vvol + 0, 1)) * sweight(vvol + 0) & ! poloidal length constraint
                - (Semn(1:mn, vvol + 1, 0)) * sweight(vvol + 1)

            IIe(vvol) = max(sum(abs(force_ft_trunc(tdoc + idoc + 1:tdoc + idoc + mn))) / (mn) , logtolerance)! screen diagnostics

            idoc = idoc + mn ! degree-of-constraint counter: increment

          end if

        end if
      else ! matches if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )

        BBe(vvol) = 9.9e+09_rkind
        IIo(vvol) = 9.9e+09_rkind
        if (.not.enforce_stell_sym) then
          BBo(vvol) = 9.9e+09_rkind
          IIe(vvol) = 9.9e+09_rkind
        end if

        force_ft_trunc(tdoc + 1:tdoc + LGdof) = 9.9e+09_rkind

      end if

    end do

    if (NGdof /= 0) then
      ForceErr = sqrt(sum(force_ft_trunc(1:NGdof) * force_ft_trunc(1:NGdof)) / NGdof)! this includes spectral constraints
    else
      ForceErr = 0.0_rkind
    end if

    if (Lhessian2Dallocated .and. Igeometry >= 3) then ! construct Hessian2D

      if (LcomputeDerivatives) then

        hessian2D(1:NGdof, 1:NGdof) = 0.0_rkind

        do vvol = 1, Mvol - 1 ! loop over interior surfaces

          if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this
            ! interface are valid

            idof = 0 ! labels degree-of-freedom = Fourier harmonic of surface geometry
            idofr = 0
            idofz = 0

            do ii = 1, mn ! loop over degrees-of-freedom

              do irz = 0, 1 ! Fourier harmonic of R, Fourier harmonic of Z

                !if( irz.eq.1 .and. Igeometry.lt.3 ) cycle ! no dependence on Z

                do issym = 0, 1 ! stellarator symmetry

                  if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on the non-stellarator symmetric harmonics

                  if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                  if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

                  idof = idof + 1 ! labels degree-of-freedom

                  !if( LocalConstraint) then

                  if (irz == 0) then
                    idofr = idofr + 1 ! labels degree-of-freedom
                  else
                    idofz = idofz + 1
                  end if
                  ! use tdof for number of volume more than 2
                  if (vvol > 1) then
                    tdofr = (vvol - 2) * LGdof + idofr ! labels degree-of-freedom in internal interface geometry
                    tdofz = (vvol - 2) * LGdof + idofz
                    tdoc = (vvol - 1) * LGdof ! labels force-balance constraint across internal interfaces
                    idoc = 0 ! local  force-balance constraint across internal interface
                    if (irz == 0) then
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdofr) = -denergydrr(idoc + 1:idoc + LGdof, vvol + 0&
                          & , 1, idof, 0)
                    else
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdofz + mn) = -denergydzr(idoc + 1:idoc + LGdof,&
                          & vvol + 0, 1, idof, 0)
                    end if
                  end if

                  tdofr = (vvol - 1) * LGdof + idofr
                  tdofz = (vvol - 1) * LGdof + idofz
                  tdoc = (vvol - 1) * LGdof ! shorthand
                  idoc = 0 ! diagonal elements
                  if (irz == 0) then
                    hessian2D(tdoc + 1:tdoc + LGdof, tdofr) = denergydrr(idoc + 1:idoc + LGdof, vvol + 1, 0, idof, 0) -&
                        & denergydrr(idoc + 1:idoc + LGdof, vvol + 0, 1, idof, 1)
                  else
                    hessian2D(tdoc + 1:tdoc + LGdof, tdofz + mn) = denergydzr(idoc + 1:idoc + LGdof, vvol + 1, 0, idof,&
                        & 0) - denergydzr(idoc + 1:idoc + LGdof, vvol + 0, 1, idof, 1)
                  end if

                  if (vvol < Mvol - 1) then
                    tdofr = (vvol + 0) * LGdof + idofr
                    tdofz = (vvol + 0) * LGdof + idofz
                    tdoc = (vvol - 1) * LGdof ! shorthand
                    idoc = 0

                    if (irz == 0) then
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdofr) = denergydrr(idoc + 1:idoc + LGdof, vvol + 1,&
                          & 0, idof, 1)
                    else
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdofz + mn) = denergydzr(idoc + 1:idoc + LGdof, vvol&
                          & + 1, 0, idof, 1)
                    end if
                  end if
                end do ! matches do issym

              end do ! matches do irz
            end do ! matches do ii

          else ! matches if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )

            call fatal(.true., __FILE__, "need to provide suitable values for hessian2D in case of field failure ")

          end if

        end do
      end if

    end if

    if (LHmatrix .and. Lhessian2Dallocated .and. Igeometry == 2) then ! construct Hessian2D

      if (LcomputeDerivatives) then

        hessian2D(1:NGdof, 1:NGdof) = 0.0_rkind

        do vvol = 1, Mvol - 1 ! loop over interior surfaces

          if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this
            ! interface are valid

            idof = 0 ! labels degree-of-freedom = Fourier harmonic of surface geometry

            do ii = 1, mn ! loop over degrees-of-freedom

              do irz = 0, 1 ! Fourier harmonic of R, Fourier harmonic of Z

                if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

                do issym = 0, 1 ! stellarator symmetry

                  if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on the non-stellarator symmetric harmonics

                  if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                  if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

                  idof = idof + 1 ! labels degree-of-freedom

                  if (LocalConstraint) then
                    ! Derivative with respect to previous interface
                    if (vvol > 1) then
                      tdof = (vvol - 2) * LGdof + idof ! labels degree-of-freedom in internal interface geometry
                      tdoc = (vvol - 1) * LGdof ! labels force-balance constraint across internal interfaces
                      idoc = 0 ! local  force-balance constraint across internal interface
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = -HdFFdRZ(idoc + 1:idoc + LGdof, 1, idof, 0,&
                          & vvol + 0)

                    end if

                    ! Derivative with respect to current interface
                    tdof = (vvol - 1) * LGdof + idof
                    tdoc = (vvol - 1) * LGdof ! shorthand
                    idoc = 0

                    hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = HdFFdRZ(idoc + 1:idoc + LGdof, 0, idof, 0,&
                        & vvol + 1) - HdFFdRZ(idoc + 1:idoc + LGdof, 1, idof, 1, vvol + 0)

                    ! Derivative with respect to next interface
                    if (vvol < Mvol - 1) then
                      tdof = (vvol + 0) * LGdof + idof
                      tdoc = (vvol - 1) * LGdof ! shorthand
                      idoc = 0
                      hessian2D(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = HdFFdRZ(idoc + 1:idoc + LGdof, 0, idof, 1,&
                          & vvol + 1)
                    end if

                    ! Case of last interface in case of boundary variation ?
                    if (vvol == Mvol - 1) then
                      !tdof = (vvol+0) * LGdof + idof
                      tdoc = (vvol - 1) * LGdof ! shorthand
                      idoc = 0
                    end if

                  else ! Global constraint

                    ! In the general case of global constraint, there are no zero element in the hessian. We thus loop
                    ! again on all volumes

                    call fatal(.true., __FILE__, "incorrect choice of Lconstraint in SPECTRE")

                  end if ! matches if( LocalConstraint )

                end do ! matches do issym

              end do ! matches do irz
            end do ! matches do ii

          else ! matches if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )

            call fatal(.true., __FILE__, "need to provide suitable values for hessian2D in case of field failure ")

          end if

        end do
      end if

    end if

    ! CONSTRUCT FORCE GRADIENT
    ! ------------------------

    if (LcomputeDerivatives .and. force_ft_trunc_grad_alloc) then ! construct Hessian
      force_ft_trunc_grad(1:NGdof, 1:NGdof) = 0.0_rkind

      do vvol = 1, Mvol - 1 ! loop over interior surfaces

        if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this
          ! interface are valid

          idof = 0 ! labels degree-of-freedom = Fourier harmonic of surface geometry

          do ii = 1, mn ! loop over degrees-of-freedom
            do irz = 0, 1 ! Fourier harmonic of R, Fourier harmonic of Z
              if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

              do issym = 0, 1 ! stellarator symmetry
                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on the non-stellarator symmetric harmonics

                if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

                idof = idof + 1 ! labels degree-of-freedom

                if (LocalConstraint) then
                  ! Derivative with respect to previous interface
                  if (vvol > 1) then
                    tdof = (vvol - 2) * LGdof + idof ! labels degree-of-freedom in internal interface geometry
                    tdoc = (vvol - 1) * LGdof ! labels force-balance constraint across internal interfaces
                    idoc = 0 ! local  force-balance constraint across internal interface
                    force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = -dFFdRZ(idoc + 1:idoc + LGdof, 1, idof, 0, &
                        vvol &
                        &+ 0)
                    if (Lconstraint == 1) then ! this is a little clumsy: could include Lfreebound or something . . .
                      force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = force_ft_trunc_grad(tdoc + idoc + 1:tdoc &
                          + idoc + LGdof,&
                          & tdof) - dBBdmp(idoc + 1:idoc + LGdof, vvol + 0, 1, 1) * dmupfdx(vvol, 1, 1, idof, 0) -&
                          & dBBdmp(idoc + 1:idoc + LGdof, vvol + 0, 1, 2) * dmupfdx(vvol, 1, 2, idof, 0)
                    end if
                  end if

                  ! Derivative with respect to current interface
                  tdof = (vvol - 1) * LGdof + idof
                  tdoc = (vvol - 1) * LGdof ! shorthand
                  idoc = 0

                  force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = dFFdRZ(idoc + 1:idoc + LGdof, 0, idof, 0, vvol +&
                      & 1) - dFFdRZ(idoc + 1:idoc + LGdof, 1, idof, 1, vvol + 0)
                  if (Lconstraint == 1) then ! this is a little clumsy
                    force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = force_ft_trunc_grad(tdoc + idoc + 1:tdoc &
                        + idoc + LGdof,&
                        & tdof) + dBBdmp(idoc + 1:idoc + LGdof, vvol + 1, 0, 1) * dmupfdx(vvol + 1, 1, 1, idof, 0) +&
                        & dBBdmp(idoc + 1:idoc + LGdof, vvol + 1, 0, 2) * dmupfdx(vvol + 1, 1, 2, idof, 0) - dBBdmp(idoc &
                        &+ 1:idoc + LGdof, vvol + 0, 1, 1) * dmupfdx(vvol + 0, 1, 1, idof, 1) - dBBdmp(idoc + 1:idoc +&
                        & LGdof, vvol + 0, 1, 2) * dmupfdx(vvol + 0, 1, 2, idof, 1)
                  end if

                  ! Derivative with respect to next interface
                  if (vvol < Mvol - 1) then
                    tdof = (vvol + 0) * LGdof + idof
                    tdoc = (vvol - 1) * LGdof ! shorthand
                    idoc = 0

                    force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = dFFdRZ(idoc + 1:idoc + LGdof, 0, idof, 1, vvol&
                        & +&
                        & 1)
                    if (Lconstraint == 1) then ! this is a little clumsy
                      force_ft_trunc_grad(tdoc + idoc + 1:tdoc + idoc + LGdof, tdof) = force_ft_trunc_grad(tdoc + idoc + 1:tdoc &
                          + idoc + LGdof,&
                          & tdof) + dBBdmp(idoc + 1:idoc + LGdof, vvol + 1, 0, 1) * dmupfdx(vvol + 1, 1, 1, idof, 1) +&
                          & dBBdmp(idoc + 1:idoc + LGdof, vvol + 1, 0, 2) * dmupfdx(vvol + 1, 1, 2, idof, 1)
                    end if
                  end if

                else ! Global constraint

                  ! In the general case of global constraint, there are no zero element in the hessian. We thus loop
                  ! again on all volumes

                  do ivol = 1, Mvol - 1
                    tdoc = (ivol - 1) * LGdof ! shorthand
                    tdof = (vvol - 1) * LGdof + idof

                    if (ivol == vvol - 1) then
                      force_ft_trunc_grad(tdoc + 1:tdoc + LGdof, tdof) = dFFdRZ(1:LGdof, 0, idof, 1, ivol + 1)
                    elseif (ivol == vvol) then
                      force_ft_trunc_grad(tdoc + 1:tdoc + LGdof, tdof) = dFFdRZ(1:LGdof, 0, idof, 0, ivol + 1) - dFFdRZ(1:LGdof, 1,&
                          & idof, 1, ivol)
                    elseif (ivol == vvol + 1) then
                      force_ft_trunc_grad(tdoc + 1:tdoc + LGdof, tdof) = -dFFdRZ(1:LGdof, 1, idof, 0, ivol)
                    end if

                    force_ft_trunc_grad(tdoc + 1:tdoc + LGdof, tdof) = force_ft_trunc_grad(tdoc + 1:tdoc + LGdof, tdof) &
                        + dBBdmp(1:LGdof, ivol +&
                        & 1, 0, 1) * dmupfdx(ivol + 1, vvol, 1, idof, 1) &
                        + dBBdmp(1:LGdof, ivol + 1, 0, 2) * dmupfdx(ivol&
                        & + 1, vvol, 2, idof, 1) - dBBdmp(1:LGdof, ivol + 0, 1, 1) * dmupfdx(ivol + 0, vvol, 1, idof, 1)&
                        & - dBBdmp(1:LGdof, ivol + 0, 1, 2) * dmupfdx(ivol + 0, vvol, 2, idof, 1)

                  end do
                end if ! matches if( LocalConstraint )
              end do ! matches do issym
            end do ! matches do irz
          end do ! matches do ii

        else ! matches if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )

          call fatal(.true., __FILE__, "need to provide suitable values for hessian in case of field failure ")

        end if
      end do
    end if

    if (force_ft_trunc_grad_alloc .and. Igeometry == 1) then
      if (myid == 0) then
        cput = mpi_wtime()
        write(ounit, '("hesian : ",f10.2," : LHmatrix="L2" ")')cput - cpus, LHmatrix
        write(ounit, *) "Writing .hessian file..."
        open(munit, file=trim(ext) // ".sp.hessian", status="unknown", form="unformatted")
        write(munit)NGdof
        write(munit)force_ft_trunc_grad(1:NGdof, 1:NGdof)
        close(munit)

      end if
    end if

    if ( LcomputeDerivatives ) then
      deallocate(dFFdRZ)
      deallocate(dBBdmp)
    end if

  end subroutine dforce

  subroutine dfp100(Ndofgl, x, Fvec, LComputeDerivatives)
    !------
    ! vvol:                       loop index on volumes
    ! Ndofgl:                     Input parameter necessary for the use of hybrd1. Unused otherwise.
    ! iflag:                      Flag changed by hybrd1
    ! cpu_send_one, cpu_send_two: CPU IDs, used for MPI communications
    ! status:                     MPI status
    ! Fvec:                       Global constraint values
    ! x:                          Degrees of freedom of hybrd1. For now contains only the poloidal flux
    integer, intent(in) :: Ndofgl
    real(rkind), intent(in) :: x(1:Mvol - 1)
    real(rkind), intent(inout) :: Fvec(1:Ndofgl)
    logical, intent(in) :: LComputeDerivatives

    integer :: vvol, iflag, cpu_send_one, cpu_send_two, ll, NN, ideriv, iocons
    real(rkind) :: Bt00(1:Mvol, 0:1, -1:2), ldItGp(0:1, -1:2)
    integer :: Lcurvature, max_bnorm_iter, iter

    dpflux(2:Mvol) = x - xoffset

    ! Now each CPU perform the calculation in its volume(s)
    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      ImagneticOK(vvol) = .false.

      ! Determines if this volume vvol should be computed by this task.
      call IsMyVolume(vvol)

      if (IsMyVolumeValue == 0) then
        cycle
      else if (IsMyVolumeValue == -1) then
        call fatal(.true., __FILE__, "Unassociated volume")

      end if

      NN = NAdof(vvol)
      ll = Lrad(vvol)

      call allocate_geometry_matrices(vvol, LcomputeDerivatives, Ntz)
      call allocate_Beltrami_matrices(vvol, LcomputeDerivatives)
      call intghs_workspace_init(vvol, Ntz)

      dBdX%L = .false. ! No need to take derivatives of matrices w.r.t geometry.

      if (update_bnorm_inbeltrami .and. (Lfreebound == 1) ) then
        max_bnorm_iter = 100
      else
        max_bnorm_iter = 1
      end if

      do iter = 1, max_bnorm_iter

        ideriv = 0
        Lcurvature = 1

        call compute_guvijsave(Iquad(vvol), vvol, ideriv, Lcurvature, Nt, Nz)
        Lsavedguvij = .true.

        call volume_integrate_chebyshev(Iquad(vvol), mn, vvol, ll, Nt, Nz)! compute volume integrals of metric elements;
        call matrix(vvol, mn, ll)

        ! Call Beltrami solver to get the magnetic field in the current volume.
        call construct_beltrami_field(vvol, NN)

        Lsavedguvij = .false.

        if (update_bnorm_inbeltrami .and. (Lfreebound == 1) ) then
          ! compute normal field etc. on computational boundary;
          if ( myid == 0 ) then
            write(ounit, *)"calculating bnorm (", iter, ")"
          end if
          call compute_bnorm_plasmab(mn, Ntz, efmn(1:mn), ofmn(1:mn))

          ! calculate error from previous Bnorm
          if (enforce_stell_sym) then
            bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1)
          else
            bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1) + sum(abs(iBnc(1:mn) - efmn(1:mn))) / (mn )
          end if

          if ( myid == 0 ) write(stderr, *)"error ", bnserr, " tol ", gBntol

          if ( bnserr > gBntol ) then
            ! update ibns, ibnc
            iBns(2:mn) = ofmn(2:mn) * ( 1.0_rkind - gBnbld ) + gBnbld * iBns(2:mn)
            if (.not.enforce_stell_sym) then
              iBnc(1:mn) = efmn(1:mn) * ( 1.0_rkind - gBnbld ) + gBnbld * iBnc(1:mn)
            end if
          else
            exit
          end if

        end if
      end do

      call intghs_workspace_destroy()
      call deallocate_Beltrami_matrices(LcomputeDerivatives)
      call deallocate_geometry_matrices(LcomputeDerivatives)

      ! Compute relevant local quantities for the evaluation of global constraints
      if (Lconstraint == 3) then
        do iocons = 0, 1
          if ( (Lcoordinatesingularity .and. iocons == 0) .or. (Lvacuumregion .and. iocons == 1) ) cycle

          ideriv = 0
          call lbpol(vvol, Mvol, Nt, Nz, Bt00(1:Mvol, 0:1, -1:2), ideriv, iocons)!Compute field at interface for global constraint

          ideriv = 2
          call lbpol(vvol, Mvol, Nt, Nz, Bt00(1:Mvol, 0:1, -1:2), ideriv, iocons)!Compute field at interface for global constraint, d
          !w.r.t. pflux
        end do

        if (Lvacuumregion) then
          ideriv = 1 ! derivatives of Btheta w.r.t tflux
          iocons = 0 ! Only need inner side of volume derivatives
          call lbpol(Mvol, Mvol, Nt, Nz, Bt00(1:Mvol, 0:1, -1:2), ideriv, iocons)

          iflag = 2 ! derivatives of poloidal linking current w.r.t geometry not required
          call compute_plasma_current(Mvol, mn, Nt, Nz, iflag, ldItGp(0:1, -1:2))
        end if
      end if
    end do

    ! Evaluation of global constraint and communications
    if (.not.LocalConstraint) then

      select case (Lconstraint)

        ! Case 3: toroidal current constraint
      case (3)

        ! Compute IPDt on each interface.
        do vvol = 1, Mvol - 1

          ! --------------------------------------------------------------------------------------------------
          !                                                                     MPI COMMUNICATIONS
          cpu_send_one = WhichCpuID(vvol, ncpu)
          cpu_send_two = WhichCpuID(vvol + 1, ncpu)

          ! Broadcast magnetic field at the interface.
          call broadcast(Bt00(vvol, 1, 0), cpu_send_one)
          call broadcast(Bt00(vvol + 1, 0, 0), cpu_send_two)
          call broadcast(Bt00(vvol, 1, 2), cpu_send_one)
          call broadcast(Bt00(vvol + 1, 0, 2), cpu_send_two)

          ! Evaluate surface current
          IPDt(vvol) = pi2 * (Bt00(vvol + 1, 0, 0) - Bt00(vvol, 1, 0))

          ! their derivatives
          IPDtdPf(vvol, vvol) = pi2 * Bt00(vvol + 1, 0, 2)
          if (vvol /= 1) IPDtdPf(vvol, vvol - 1) = -pi2 * Bt00(vvol, 1, 2)
        end do

        ! Compute the constraint and store it in Fvec.
        if (myid == 0) then
          Fvec(1:Mvol - 1) = IPDt(1:Mvol - 1) - Isurf(1:Mvol - 1)
        end if

        ! Set the total pflux to 0
        if (Igeometry == 1) then

          IPDtDpf(1, Mvol) = -pi2 * Bt00(1, 1, 2)
          IPdtdPf(2:Mvol, Mvol) = 0.0_rkind
          IPDtDpf(Mvol, 1:Mvol) = 1.0_rkind

          ! Constraint the total pflux (pflux(Mvol))
          ! zero for symmetric initial profiles
          ! non-zero for asymmetric profiles
          Fvec(Mvol) = sum(dpflux(1:Mvol)) - total_pflux
        end if

        ! Compute poloidal linking current constraint as well in case of free boundary computation
        if (Lfreebound == 1) then

          ! Communicate additional derivatives
          cpu_send_one = WhichCpuID(Mvol, ncpu)
          call broadcast(ldItGp(0:1, -1:2), 8, cpu_send_one)
          call broadcast(Bt00(Mvol, 0:1, 1), 2, cpu_send_one)

          ! Complete output: RHS
          Fvec(Mvol) = ldItGp(1, 0) - curpol

          ! Complete output: LHS
          IPDtdPf(Mvol - 1, Mvol) = pi2 * Bt00(Mvol, 0, 1)
          IPDtdPf(Mvol, Mvol - 1) = ldItGp(1, 2)
          IPDtdPf(Mvol, Mvol) = ldItGp(1, 1)
        end if

      case default
        call fatal(.true., __FILE__, "Unaccepted value for Lconstraint")

      end select
    end if

  end subroutine dfp100

  subroutine dfp200(LcomputeDerivatives)
    logical, intent(in) :: LComputeDerivatives ! indicates whether derivatives are to be calculated

    logical :: LInnerVolume
    integer :: ierr
    integer :: NN, vflag, lvol, vvol
    integer :: innout, ii, jj, irz, issym, iocons, idof, ll
    integer :: ideriv, id
    integer :: iflag, cpu_id, cpu_id1, even_or_odd, vol_parity
    integer :: tag, tag2
    MPI_STATUS_TYPE :: stat

    real(rkind) :: DDl, MMl
    real(rkind), allocatable :: XX(:), YY(:), dBB(:, :), dII(:), dLL(:), dPP(:), length(:), dRR(:, :), dZZ(:, :),&
        & constraint(:)
    real(rkind), allocatable :: ddFcol1(:), ddFcol2(:), ddFcol3(:), ddFcol4(:)

    character :: packorunpack

    type(MatrixLU) :: oBI(1:Mvol)

    call sallocate(dBB, 1, Ntz, -1, 2, 0.0_rkind)! magnetic field strength (on interfaces) in real space and derivatives
    call sallocate(XX, 1, Ntz, 0.0_rkind)
    call sallocate(YY, 1, Ntz, 0.0_rkind)
    call sallocate(length, 1, Ntz, 0.0_rkind)! this is calculated in lforce

    if (LComputeDerivatives) then
      call sallocate(dRR, 1, Ntz, -1, 1, 0.0_rkind)
      call sallocate(dZZ, 1, Ntz, -1, 1, 0.0_rkind)
      call sallocate(dII, 1, Ntz, 0.0_rkind)! spectral constraint
      call sallocate(dLL, 1, Ntz, 0.0_rkind)! length   constraint
      call sallocate(dPP, 1, Ntz, 0.0_rkind)! poloidal constraint
      call sallocate(constraint, 1, Ntz, 0.0_rkind)
      call sallocate(ddFcol1, 1, Ntz, 0.0_rkind)
      call sallocate(ddFcol2, 1, Ntz, 0.0_rkind)
      call sallocate(ddFcol3, 1, Ntz, 0.0_rkind)
      call sallocate(ddFcol4, 1, Ntz, 0.0_rkind)
    end if

    if (LocalConstraint) then

      do vvol = 1, Mvol

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")

        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

        dBdX%vol = vvol ! Label
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz)! compute volume

        !!----Hessian2D cleared-----

        do iocons = 0, 1 ! construct field magnitude on inner and outer interfaces: inside do vvol

          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     : there are no constraints
          ! at outer boundary

          ideriv = 0
          id = ideriv
          iflag = 0 ! XX & YY are returned by lforce: Bemn(1:mn,vvol,iocons), Iomn(1:mn,vvol) etc. are returned through
          ! global
          call lforce(vvol, iocons, ideriv, Ntz, dBB(1:Ntz, id), XX(1:Ntz), YY(1:Ntz), length(1:Ntz), DDl, MMl, iflag)

        end do

        if (LcomputeDerivatives) then ! compute inverse of Beltrami matrices

          ! Allocate some memory for storing the LU matrix and ipivot
          ! we need 1.0_rkind extra dimension for helicity constraint, so dimension is NN+1 -> 0:NN
          call sallocate(oBI(vvol)%mat, 0, NN, 0, NN, 0.0_rkind)! inverse of ``original'', i.e. unperturbed, Beltrami
          ! matrix
          call sallocate(oBI(vvol)%ipivot, 0, NN, 0)

          ! initialize matrices for Beltrami linear system
          call allocate_geometry_matrices(vvol, LcomputeDerivatives, Ntz)
          call allocate_Beltrami_matrices(vvol, LcomputeDerivatives)
          call intghs_workspace_init(vvol, Ntz)

          packorunpack = 'P'
          call packab(packorunpack, vvol, NN, solution(1:NN, 0), 0)! packing, put the solutions back to the solution
          ! matrix

          ! Invert beltrami matrix. Required for matrix perturbation theory
          call get_LU_Beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)

          dBdX%L = .true. ! will need derivatives
          idof = 0 ! labels degree of freedom: local to interface

          do ii = 1, mn ! loop over deformations in Fourier harmonics: inside do vvol

            dBdX%ii = ii ! controls construction of derivatives in subroutines called below
            do irz = 0, 1 ! loop over deformations in R and Z: inside do vvol: inside do ii

              if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
              dBdX%irz = irz ! controls construction of derivatives

              do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms

                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

                dBdX%issym = issym ! controls construction of derivatives
                idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to: this is
                ! outside do innout

                do innout = 0, 1 ! loop over deformations to inner and outer interface: inside do vvol: inside do ii:
                  ! inside do irz

                  if (vvol == 1 .and. innout == 0) cycle ! no degrees of freedom at coordinate axis / fixed inner boundary
                  if (vvol == Mvol .and. innout == 1) cycle ! no degress of freedom                      fixed outer
                  ! boundary: for linearized displacement

                  dBdX%innout = innout

                  ! Evaluate derivatives of field w.r.t geometry
                  call get_perturbed_solution(vvol, oBI(vvol), NN, Nt, Nz)

                  ! Helicity multiplier and poloidal flux derivatives w.r.t geometry
                  call evaluate_dmupfdx(innout, idof, ii, Nt, Nz)

                  !-----Hessian 2D cleared

                  ! Evaluate derivatives of B square
                  if (force_ft_trunc_grad_alloc) then
                    call evaluate_dBB(vvol, idof, innout, issym, irz, ii, Ntz, dBB, XX, YY, length, dRR, dZZ, dII, dLL, dPP,&
                        & LComputeDerivatives)
                  end if

                  if (Lhessian2Dallocated .and. Igeometry.ge.3) then
                    call hessian3D_dFFdRZ(vvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dRR, dZZ, dII, dLL, dPP, Ntz)
                  endif
                end do ! matches do innout
              end do ! matches do issym
            end do ! matches do irz
          end do ! matches do ii

          dBdX%L = .false. ! probably not needed, but included anyway

          call intghs_workspace_destroy()
          call deallocate_Beltrami_matrices(LcomputeDerivatives)
          call deallocate_geometry_matrices(LcomputeDerivatives)

          deallocate(oBI(vvol)%mat)
          deallocate(oBI(vvol)%ipivot)

        end if

      end do ! matches do vvol = 1, Mvol

    else ! CASE SEMI GLOBAL CONSTRAINT

      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")

        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz)! compute volume

        do iocons = 0, 1 ! construct field magnitude on inner and outer interfaces: inside do vvol

          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     : there are no constraints
          ! at outer boundary

          ideriv = 0
          id = ideriv
          iflag = 0 ! dAt, dAz, XX & YY are returned by lforce: Bemn(1:mn,vvol,iocons), Iomn(1:mn,vvol) etc. are returned
          ! through global
          call lforce(vvol, iocons, ideriv, Ntz, dBB(1:Ntz, id), XX(1:Ntz), YY(1:Ntz), length(1:Ntz), DDl, MMl, iflag)

        end do
      end do

      if (LComputeDerivatives) then
        dBdX%L = .true. ! will need derivatives

        ! Each volume will need solution and derivative information in vacuum region. Thus, broadcast!
        ! Derivative of w.r.t geometry only required if plasma interface is perturbed - this is
        ! calculated below
        if (Lfreebound == 1) then
          do ideriv = 0, 2
            do ii = 1, mn

              cpu_id = WhichCpuID(Mvol, ncpu)

              call broadcast(Ate(Mvol, ideriv, ii)%s(0:Lrad(Mvol)), Lrad(Mvol) + 1, cpu_id)
              call broadcast(Aze(Mvol, ideriv, ii)%s(0:Lrad(Mvol)), Lrad(Mvol) + 1, cpu_id)
            end do
          end do
        end if

        ! Evaluation of froce gradient in case of semi-global constraint (for now, this is specific to Lconstraint=3)
        ! Loop over all geometrical degrees of freedom. Thi means we loop over
        ! - each interface (vvol)
        ! - each Fourier mode (ii)
        ! - on R or Z (irz)
        ! - on symmetric and non stellarator symmetric terms (issym)

        ! Then the perturbed solution is evaluated on volumes sharing the interface vvol (volume vvol and vvol+1).
        ! i.e. relevant routines are run with lvol=vvol, innout=1 and lvol=vvol+1, innout=0

        ! First invert Beltrami matrices and store them in OBI
        do vvol = 1, Mvol
          call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
          NN = NAdof(vvol)! shorthand
          ll = Lrad(vvol)

          call sallocate(oBI(vvol)%mat, 0, NN, 0, NN, 0.0_rkind)! inverse of ``original'', i.e. unperturbed, Beltrami
          ! matrix
          call sallocate(oBI(vvol)%ipivot, 0, NN, 0)

          ! Parallelization
          call IsMyVolume(vvol)
          if (IsMyVolumeValue == 0) then
            cycle
          else if (IsMyVolumeValue == -1) then
            call fatal(.true., __FILE__, "Unassociated volume")

          end if

          ! Invert LHS of Beltrami system and store it in oBI. This will be used to
          ! evaluate derivatives of solution.
          ! TODO: Storage of OBi is not optimal. Find a way around?

          call allocate_geometry_matrices(vvol, LcomputeDerivatives, Ntz)
          call allocate_Beltrami_matrices(vvol, LcomputeDerivatives)

          call get_LU_beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)

          call deallocate_Beltrami_matrices(LcomputeDerivatives)
          call deallocate_geometry_matrices(LcomputeDerivatives)
        end do

        ! Broadcast oBI to all CPU
        ! TODO: THIS is bad! oBI can be very large... Change parallelization?
        !       Another possible parallelization: Each CPU keep derivatives info about its solution,
        !                                         CPU1 build linear system and solve it?
        if (ncpu > 1) then
          do vvol = 1, Mvol
            NN = NAdof(vvol)
            cpu_id = WhichCpuID(vvol, ncpu)
            call broadcast(oBI(vvol)%mat(0:NN, 0:NN), (NN + 1)  ** 2, cpu_id)
            call broadcast(oBI(vvol)%ipivot(0:NN), NN + 1, cpu_id)
          end do
        end if

        ! Loop on perturbed interfaces
        do even_or_odd = 0, 1 ! First loop on even interfaces perturbation, then on odd interfaces. This allow efficient
          ! parallelization

          do vvol = 1, Mvol - 1 !labels which interface is perturbed

            ! Parallelization - parallelization on perturbed interface and not on volume (outermost loop)
            vol_parity = modulo(vvol, 2)
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

            dBdX%vol = vvol ! Perturbed interface
            idof = 0 ! labels degree of freedom of interface vvol

            do ii = 1, mn ! loop over deformations in Fourier harmonics: inside do vvol
              dBdX%ii = ii ! controls construction of derivatives in subroutines called below

              do irz = 0, 1 ! loop over deformations in R and Z: inside do vvol: inside do ii
                if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
                dBdX%irz = irz ! controls construction of derivatives

                do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
                  if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                  if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                  if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}
                  dBdX%issym = issym ! controls construction of derivatives

                  idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to: this
                  ! is outside do innout

                  do lvol = vvol, vvol + 1

                    ! Only interested in 1.0_rkind interface perturbation
                    if (ncpu > 1) then
                      if (LinnerVolume .and. (lvol == vvol + 1) ) cycle
                      if (.not.LinnerVolume .and. (lvol == vvol) ) cycle
                    end if

                    ! Set up perturbation information
                    if (lvol == vvol) innout = 1 ! Perturb w.r.t outer interface
                    if (lvol == vvol + 1) innout = 0 ! Perturb w.r.t inner interface

                    dBdX%innout = innout
                    dBdX%L = .true.

                    ! Set up volume information
                    call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
                    NN = NAdof(lvol)

                    ! Allocate memory. This cannot be moved outside due to NN and ll dependence on volume.
                    call allocate_geometry_matrices(lvol, LcomputeDerivatives, Ntz)
                    call allocate_Beltrami_matrices(lvol, LcomputeDerivatives)

                    ! Get derivative of vector potential w.r.t geometry. Matrix perturbation theory.
                    call intghs_workspace_init(lvol, Ntz)
                    call get_perturbed_solution(lvol, oBI(lvol), NN, Nt, Nz)
                    call intghs_workspace_destroy()

                    ! Free memory
                    call deallocate_Beltrami_matrices(LcomputeDerivatives)
                    call deallocate_geometry_matrices(LcomputeDerivatives)

                  end do

                  ! Now volumes neighbouring the interface share perturbed solution
                  cpu_id = WhichCpuID(vvol, ncpu)
                  cpu_id1 = WhichCpuID(vvol + 1, ncpu)

                  ! TODO IMPROVE MPI COMMUNICATIONS. Could directly send dAt and dAz?
                  ! Gather everything in inner volume
                  if (ncpu > 1) then
                    if (LinnerVolume) then
                      do jj = 1, mn
                        tag = 1 ! Tags for MPI communications
                        tag2 = 2

                        call MPI_RECV(Ate(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION&
                            & , cpu_id1, tag, MPI_COMM_SPEC, stat, ierr)
                        call MPI_RECV(Aze(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION&
                            & , cpu_id1, tag2, MPI_COMM_SPEC, stat, ierr)
                      end do

                      ! Non-stellarator symmetric terms
                      if (.not.enforce_stell_sym) then
                        do jj = 1, mn
                          tag = 3
                          tag2 = 4

                          call MPI_RECV(Ato(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1,&
                              & MPI_DOUBLE_PRECISION, cpu_id1, tag, MPI_COMM_SPEC, stat, ierr)
                          call MPI_RECV(Azo(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1,&
                              & MPI_DOUBLE_PRECISION, cpu_id1, tag2, MPI_COMM_SPEC, stat, ierr)
                        end do
                      end if

                    else
                      do jj = 1, mn
                        tag = 1
                        tag2 = 2

                        call MPI_SEND(Ate(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION&
                            & , cpu_id, tag, MPI_COMM_SPEC, ierr)
                        call MPI_SEND(Aze(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION&
                            & , cpu_id, tag2, MPI_COMM_SPEC, ierr)
                      end do

                      if (.not.enforce_stell_sym) then
                        do jj = 1, mn
                          tag = 3
                          tag2 = 4

                          call MPI_SEND(Ato(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1,&
                              & MPI_DOUBLE_PRECISION, cpu_id, tag, MPI_COMM_SPEC, ierr)
                          call MPI_SEND(Azo(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1,&
                              & MPI_DOUBLE_PRECISION, cpu_id, tag2, MPI_COMM_SPEC, ierr)
                        end do
                      end if
                    end if
                  end if

                  ! At this point, the inverted original Beltrami matrix oBI and the perturbed solution for each volume
                  ! neighboring the perturbed interface in inner volume cpu. We now compute the derivatives of mu and
                  ! psip w.r.t the position and dBB.
                  if (LinnerVolume) then
                    ! Helicity multiplier and poloidal flux derivatives
                    call evaluate_dmupfdx(1, idof, ii, Nt, Nz)
                  else
                    ! Only inner volume computes the derivatives in all volumes (it will be broadcasted later)
                    dmupfdx(1:Mvol, vvol, 1:2, idof, 1) = 0.0_rkind
                  end if

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
                    dBdX%innout = innout

                    call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

                    ! EVALUATE dBB
                    call evaluate_dBB(lvol, idof, innout, issym, irz, ii, Ntz, dBB, XX, YY, length, dRR, dZZ, dII, dLL, dPP,&
                        & LComputeDerivatives)

                    if (Lhessian2Dallocated .and. Igeometry.ge.3) then
                      call hessian3D_dFFdRZ(vvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dRR, dZZ, dII, dLL, dPP, Ntz)
                    endif
                  end do ! matches do lvol = vvol, vvol+1

                end do ! matches do issym
              end do ! matches do irz
            end do ! matches do ii
          end do ! matches do vvol
        end do ! matches do even_or_odd

        ! Free memory
        do vvol = 1, Mvol
          deallocate(oBi(vvol)%mat)
          deallocate(oBI(vvol)%ipivot)
        end do

        dBdX%L = .false. ! probably not needed, but included anyway

        ! We know need to broadcast the vectors dmupfdx and dFFdRZ and dBBdmp
        do vvol = 1, Mvol
          cpu_id = WhichCpuID(vvol, ncpu)

          if (vvol /= Mvol) then
            call broadcast(dmupfdx(1:Mvol, vvol, 1:2, 1:LGdof, 1), Mvol * 2 * LGdof, cpu_id)
          end if

          call broadcast(dFFdRZ(1:LGdof, 0:1, 1:LGdof, 0:1, vvol), 2 * 2 * (LGdof ** 2) , cpu_id)
          call broadcast(dBBdmp(1:LGdof, vvol, 0:1, 1:2), 2 * 2 * LGdof, cpu_id)
        end do

        if (Lhessian2Dallocated) then
          do vvol = 1, Mvol
            cpu_id = WhichCpuID(vvol, ncpu)

            call broadcast(denergydrr(1:LGdof, vvol, 0:1, 1:LGdof, 0:1), 2 * 2 * (LGdof**2), cpu_id)
            call broadcast(denergydzr(1:LGdof, vvol, 0:1, 1:LGdof, 0:1), 2 * 2 * (LGdof**2), cpu_id)
          end do
        end if

      end if ! End of if( LComputeDerivatives )
    end if ! End of if( LocalConstraint )

    if (LcomputeDerivatives) then
      deallocate(constraint)
      deallocate(dPP)
      deallocate(dLL)
      deallocate(dII)
      deallocate(dZZ)
      deallocate(dRR)
      deallocate(ddFcol1)
      deallocate(ddFcol2)
      deallocate(ddFcol3)
      deallocate(ddFcol4)
    end if

    deallocate(dBB)
    deallocate(XX)! spectral constraints
    deallocate(YY)
    deallocate(length)
  end subroutine dfp200

  subroutine get_LU_beltrami_matrices(vvol, oBI, NN, nt_loc, nz_loc)

    ! Evaluate the LU factorization of Beltrami matrices and store the original one in oBI.

    ! INPUT
    ! -----
    !     vvol:     Volume number
    !   NN:        is equal to NAdof(vvol), is a shorthand.

    ! MODULES
    ! -------
    real(rkind) :: cput
    ! ------
    type(MatrixLU), intent(inout) :: oBI
    integer, intent(in) :: vvol, NN, nt_loc, nz_loc

    integer :: IA, MM, LDA, Lwork, ll
    integer :: idgetrf
    real(rkind) :: lastcpu

    lastcpu = MPI_Wtime()

    ll = Lrad(vvol)

    ! Evaluate dMa, dMD
    dBdX%L = .false.

    Lsavedguvij = .false.

    call volume_integrate_chebyshev(Iquad(vvol), mn, vvol, ll, Nt, Nz)

    call matrix(vvol, mn, ll)

    lastcpu = MPI_Wtime()

    if (Lconstraint == 2) then ! for helicity constraint

      !dMA(1:NN,1:NN) = dMA(1:NN,1:NN) - mu(vvol) * dMD(1:NN,1:NN) ! this corrupts dMA, but dMA is no longer used
      call DAXPY( (NN + 1) * (NN + 1) , -mu(vvol), dMD, 1, dMA, 1)! BLAS version
      dMA(0, 0) = 0.0_rkind

      dMA(1:NN, 0) = -matmul(dMD(1:NN, 1:NN), solution(1:NN, 0))
      dMA(0, 1:NN) = dMA(1:NN, 0)! This is the transpose of the above

      IA = NN + 1 + 1
      MM = NN
      LDA = NN + 1
      Lwork = NN + 1
      idgetrf = 1
      call DGETRF(MM + 1, NN + 1, dMA(0:LDA - 1, 0:NN), LDA, oBI%ipivot, idgetrf)

      cput = MPI_Wtime()
      select case (idgetrf)
      case (:-1)
        write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "input error"
      case (0)
        if (Wdforce) write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "success"
      case (1:)
        write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "singular"
      case default
        call fatal(.true., __FILE__, "illegal ifail returned from F07ADF ")

      end select

      call DCOPY( (1 + NN) * (1 + NN) , dMA, 1, oBI%mat, 1)! BLAS version

    else ! for other constraints

      !dMA(0:NN-1,1:NN) = dMA(1:NN,1:NN) - mu(vvol) * dMD(1:NN,1:NN) ! this corrupts dMA, but dMA is no longer used
      !dMA(  NN  ,1:NN) = 0.0_rkind
      call DAXPY( (NN + 1) * (NN + 1) , -mu(vvol), dMD, 1, dMA, 1)! BLAS version

      !dMD(1:NN  ,1:NN) = dMA(0:NN-1,1:NN) ! copy of original matrix: this is used below
      call DCOPY( (NN + 1) * (NN + 1) , dMA, 1, dMD, 1)! BLAS version

      IA = NN + 1

      MM = NN
      LDA = NN + 1
      Lwork = NN

      idgetrf = 1
      call DGETRF(MM, NN, dMA(1, 1), LDA, oBI%ipivot(1:NN), idgetrf)

      cput = MPI_Wtime()
      select case (idgetrf)
      case (:-1)
        write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "input error"
      case (0)
        if (Wdforce) write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "success"
      case (1:)
        write(ounit, 1010)cput - cpus, myid, vvol, cput - lastcpu, idgetrf, "singular"
      case default
        call fatal(.true., __FILE__, "illegal ifail returned from F07ADF ")

      end select

      oBI%mat(1:NN, 1:NN) = dMA(1:NN, 1:NN)

    end if

    1010 format("dforce : ", f10.2, " : myid=", i3, " ; vvol=", i3, " ; called DGETRF ; time=", f10.2, &
        "s ; LU factorization of matrix; idgetrf=", i2, " ", a18)
  end subroutine get_LU_beltrami_matrices

  subroutine get_perturbed_solution(vvol, oBI, NN, nt_loc, nz_loc)

    ! This routine evaluates the value of the magnetic field once the interface is perturbed using matrix perturbation
    ! theory.
    ! Separated from the main dfp200 core to allow local and semi-global constraints to be prescribed
    !
    ! Attention: Solution is perturbed for a given degree of freedom (information stored in dBdX)
    ! use precisions_mod, only: rkind
    integer, intent(in) :: vvol, NN, nt_loc, nz_loc
    type(MatrixLU), intent(inout) :: oBI

    integer :: ll, idgetrs
    real(rkind) :: work(1:NN + 1), rhs(0:NN), dVA(0:NN) !, dVD(0:NN)
    character :: packorunpack

    ll = Lrad(vvol)! Shorthand
    dBdX%L = .true.

    call intghs(Iquad(vvol), mn, vvol, ll, 0, nt_loc, nz_loc)

    call mtrxhs(vvol, mn, ll, dVA, 0)

    rhs(0) = 0.0_rkind
    rhs(1:NN) = -dVA(1:NN)

    if (Lconstraint == 2) then
      work(1:NN + 1) = rhs(0:NN)
      !work(1:NN+1)  =  matmul( oBI(0:NN,0:NN), rhs(0:NN)) ! original version
      call DGETRS('N', NN + 1, 1, oBI%mat(0, 0), NN + 1, oBI%ipivot(0:NN), work(1), NN + 1, idgetrs)! Change to DGETRS
      solution(1:NN, -1) = work(2:NN + 1)
    else
      !solution(1:NN,-1) = matmul( oBI(1:NN,1:NN), rhs(1:NN) ) ! original version
      solution(1:NN, -1) = rhs(1:NN)
      call DGETRS('N', NN, 1, oBI%mat(1, 1), NN + 1, oBI%ipivot(1:NN), solution(1, -1), NN, idgetrs)! Change to DGETRS
    end if

    ! Unpack derivatives of solution
    packorunpack = 'U'
    call packab(packorunpack, vvol, NN, solution(1:NN, -1), -1)! derivatives placed in Ate(lvol,ideriv,1:mn)%s(0:Lrad),

  end subroutine get_perturbed_solution

  subroutine evaluate_dmupfdx(innout, idof, ii, nt_loc, nz_loc)

    ! Evaluate mu and psip derivatives and store them in dmupfdx.
    ! If debug and Lcheck=4, compare with finite difference approximation
    !
    ! INPUT:
    ! ------
    !     innout: inner or outer boundary. Not used in case of Lconstraint=3 since inside loop
    !     idof:      ???
    !     ii:     index of Fourier harmonics, ii=1:mn
    integer, intent(in) :: innout, idof, ii, nt_loc, nz_loc

    integer :: vvol, iflag, ll, NN
    integer :: vflag, pvol, order, IDGESV
    integer :: iocons
    integer, allocatable :: IPIV(:)
    real(rkind) :: det, lfactor, Bt00(1:Mvol, 0:1, -1:2)
    real(rkind), allocatable :: dBdmpf(:, :), dBdx2(:)

    vvol = dBdX%vol ! shorthand

    ll = Lrad(vvol)! shorthand
    NN = NAdof(vvol)! shorthand

    lfactor = psifactor(ii, vvol - 1 + innout)! this "pre-conditions" the geometrical degrees-of-freedom

    if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lvacuumregion .and. Lconstraint >= 0) ) then ! will need to
      ! accommodate constraints

      if (Lconstraint == 3) then

        ! In case of freeboundary, there is an additional equation in the linear system, related to the poloidal linking
        ! current.
        if (Lfreebound == 1) then
          order = Mvol
        else
          order = Mvol - 1
        end if

        call sallocate(dBdmpf, 1, order, 1, order, 0.0_rkind)
        call sallocate(dBdx2, 1, order, 0.0_rkind)
        call sallocate(IPIV, 1, order, 0)

        ! Derivatives of helicity multipliers
        dmupfdx(1:Nvol, vvol, 1, idof, 1) = 0.0_rkind ! The helicity multiplier is not varied (constrained). However
        ! dtflux is varied in vacuum region, if free boundary

        ! Derivatives of poloidal flux. Need to solve a linear system of 5 equations - by hand for now
        ! Matrix coefficients evaluation
        do pvol = 1, Mvol
          call lregion(pvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

          do iocons = 0, 1
            if ( (Lcoordinatesingularity .and. iocons == 0) .or. (Lvacuumregion .and. iocons == 1) ) cycle
            call lbpol(pvol, Mvol, nt_loc, nz_loc, Bt00(1:Mvol, 0:1, -1:2), 2, iocons)! Stores derivative in global variable Btemn
          end do
        end do

        dBdmpf(1:order, 1:order) = 0.0_rkind ! Initialize. TODO: useless?
        do pvol = 1, Mvol - 2
          dBdmpf(pvol, pvol) = Bt00(pvol + 1, 0, 2)
          dBdmpf(pvol + 1, pvol) = -Bt00(pvol + 1, 1, 2)
        end do
        dBdmpf(Mvol - 1, Mvol - 1) = Bt00(Mvol, 0, 2)

        ! RHS coefficients evaluation
        do pvol = vvol, vvol + 1
          call lregion(pvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
          if (pvol == vvol) then
            dBdX%innout = 1 ! take derivative w.r.t outer interface
          else !pvol.eq.vvol+1
            dBdX%innout = 0 ! w.r.t inner interface
          end if

          do iocons = 0, 1
            if ( (Lcoordinatesingularity .and. iocons == 0) .or. (Lvacuumregion .and. iocons == 1) ) cycle
            call lbpol(pvol, Mvol, nt_loc, nz_loc, Bt00(1:Mvol, 0:1, -1:2), -1, iocons)! derivate w.r.t geometry
          end do
        end do

        dBdx2(1:Mvol - 1) = 0.0_rkind
        if (vvol > 1) then
          dBdx2(vvol - 1) = -Bt00(vvol, 0, -1)
        end if
        dBdx2(vvol) = Bt00(vvol, 1, -1) - Bt00(vvol + 1, 0, -1)
        if (vvol < Mvol - 1) then
          dBdx2(vvol + 1) = Bt00(vvol + 1, 1, -1)
        end if

        if (Lfreebound == 1) then ! Need to modify last two equations

          ! Set all last column to zero - maybe not necessary ?
          dBdmpf(1:Mvol, Mvol) = 0.0_rkind

          ! Get derivatives of B_theta w.r.t the toroidal flux in vacuum region
          iocons = 0
          call lbpol(Mvol, Mvol, nt_loc, nz_loc, Bt00(1:Mvol, 0:1, -1:2), 1, iocons)

          ! compute d(Itor,Gpol)/dpsip and d(Itor,Gpol)/dpsit
          ! TODO: this should already be evaluated in solve_beltrami_system...
          ! TODO: THIS COULD BE MOVED OUTSIDE THE LOOPS
          iflag = 2
          call compute_plasma_current(Mvol, mn, nt_loc, nz_loc, iflag, dItGpdxtp(0:1, -1:2, Mvol))

          dBdmpf(Mvol - 1, Mvol) = Bt00(Mvol, 0, 1)!dBdpsit
          dBdmpf(Mvol, Mvol - 1) = dItGpdxtp(1, 2, Mvol)!dIpdpsip
          dBdmpf(Mvol, Mvol) = dItGpdxtp(1, 1, Mvol)!dIpdpsit

          ! compute d(Itor,Gpol)/dx
          if (vvol == Mvol - 1) then ! Plasma interface is perturbed
            iflag = -1
            call compute_plasma_current(Mvol, mn, nt_loc, nz_loc, iflag, dItGpdxtp(0:1, -1:2, Mvol))
            dBdx2(Mvol) = -dItGpdxtp(1, -1, Mvol)!-dIpdxj
          else ! Inner interface is perturbed
            dBdx2(Mvol) = 0.0_rkind
          end if
        end if

        ! Solve linear system dBdmpf * x = dBdx2. Solution (x) is stored in dBdx2 on exit
        call DGESV(order, 1, dBdmpf(1:order, 1:order), order, IPIV, dBdx2(1:order), order, IDGESV)

        ! Fill dmupfdx array with solution and multiply by coordinate conditioning factor
        dmupfdx(1, vvol, 2, idof, 1) = 0.0_rkind ! First poloidal flux is always zero
        dmupfdx(2:Mvol, vvol, 2, idof, 1) = lfactor * dBdx2(1:Mvol - 1)! These are the derivatives of pflux

        if (Lfreebound == 1) then
          dmupfdx(Mvol, vvol, 1, idof, 1) = lfactor * dBdx2(Mvol)! This is the derivative of tflux
        end if

        ! Free memory
        deallocate(dBdmpf)
        deallocate(dBdx2)
        deallocate(IPIV)

      else ! LocalConstraint

        if (Lconstraint == 1) then
          iflag = -1
          call compute_rotational_transform(vvol, mn, lmns, nt_loc, nz_loc, iflag, diotadxup(0:1, -1:2, vvol))! compute d(transform)/dx
        end if

        if (Lvacuumregion .and. Lconstraint >= 0) then
          iflag = -1
          call compute_plasma_current(vvol, mn, nt_loc, nz_loc, iflag, dItGpdxtp(0:1, -1:2, vvol))! compute d(Itor,Gpol)/dx
        end if

        dmupfdx(vvol, 1, 1, idof, innout) = 0.0_rkind ! Prepare array
        dmupfdx(vvol, 1, 2, idof, innout) = 0.0_rkind ! Prepare array

        if (Lplasmaregion) then

          if (Lconstraint == 1) then

            if (Lcoordinatesingularity) then ! solution does not depend on dpflux, and only outer transform is a
              ! constraint

              det = diotadxup(1, 1, vvol)
              call fatal(abs(det) < small, __FILE__, "error computing derivatives of mu          wrt geometry at fixed   &
                  &              transform ")

              dmupfdx(vvol, 1, 1, idof, innout) = -lfactor * (diotadxup(1, -1, vvol)) / det
              dmupfdx(vvol, 1, 2, idof, innout) = 0.0_rkind

            else ! if( .not.Lcoordinatesingularity )

              det = diotadxup(0, 1, vvol) * diotadxup(1, 2, vvol) - diotadxup(0, 2, vvol) * diotadxup(1, 1, vvol)
              call fatal(abs(det) < small, __FILE__, "error computing derivatives of mu & dpflux wrt geometry at fixed   &
                  &              transform ")

              dmupfdx(vvol, 1, 1, idof, innout) = -lfactor * (+diotadxup(1, 2, vvol) * diotadxup(0, -1, vvol) -&
                  & diotadxup(0, 2, vvol) * diotadxup(1, -1, vvol)) / det
              dmupfdx(vvol, 1, 2, idof, innout) = -lfactor * (-diotadxup(1, 1, vvol) * diotadxup(0, -1, vvol) +&
                  & diotadxup(0, 1, vvol) * diotadxup(1, -1, vvol)) / det

            end if

          end if

        else ! Vacuum region

          if (Lconstraint == 0) then ! THIS NEEDS ATTENTION

            det = dItGpdxtp(0, 1, vvol) * dItGpdxtp(1, 2, vvol) - dItGpdxtp(0, 2, vvol) * dItGpdxtp(1, 1, vvol)
            call fatal(abs(det) < small, __FILE__, "error computing derivatives of dtflux & dpflux wrt geometry at fixed &
                &              Itor and Gpol ")

            dmupfdx(vvol, 1, 1, idof, innout) = -lfactor * (+dItGpdxtp(1, 2, vvol) * dItGpdxtp(0, -1, vvol) - dItGpdxtp(0&
                &, 2, vvol) * dItGpdxtp(1, -1, vvol)) / det
            dmupfdx(vvol, 1, 2, idof, innout) = -lfactor * (-dItGpdxtp(1, 1, vvol) * dItGpdxtp(0, -1, vvol) + dItGpdxtp(0&
                &, 1, vvol) * dItGpdxtp(1, -1, vvol)) / det

          else if (Lconstraint == 1) then

            det = diotadxup(0, 1, vvol) * dItGpdxtp(1, 2, vvol) - diotadxup(0, 2, vvol) * dItGpdxtp(1, 1, vvol)
            call fatal(abs(det) < small, __FILE__, "error computing derivatives of dtflux & dpflux wrt geometry at fixed &
                &              Itor and Gpol ")

            dmupfdx(vvol, 1, 1, idof, innout) = -lfactor * (+dItGpdxtp(1, 2, vvol) * diotadxup(0, -1, vvol) - diotadxup(0&
                &, 2, vvol) * dItGpdxtp(1, -1, vvol)) / det
            dmupfdx(vvol, 1, 2, idof, innout) = -lfactor * (-dItGpdxtp(1, 1, vvol) * diotadxup(0, -1, vvol) + diotadxup(0&
                &, 1, vvol) * dItGpdxtp(1, -1, vvol)) / det

          end if

        end if

      end if

    end if

    vflag = 1 ! this flag instructs volume to continue even if the volume is invalid
    call volume(vvol, vflag, nt_loc, nz_loc)! compute derivative of volume: wrt to harmonic described by dBdX structure
  end subroutine evaluate_dmupfdx

  subroutine evaluate_dBB(lvol, idof, innout, issym, irz, ii, Ntz, dBB, XX, YY, length, dRR, dZZ, dII, dLL, dPP, &
      LcomputeDerivatives)

    ! Evaluate the derivative of the square of the magnetic field modulus. Add spectral constraint derivatives if
    ! required.

    ! INPUT
    ! -----
    !    lvol:     Volume number
    !   idof:     Labels degree of freedom
    !   innout: Index for loop on inner / outer interface (which interface is perturbed)
    !   issym:  Index for loop on stellarator non-symmetric terms
    !   irz:     Index for loop on R or Z modes
    !   ii:     Index for loop on Fourier mode
    integer, intent(in) :: lvol
    integer, intent(in) :: idof
    integer, intent(in) :: innout
    integer, intent(in) :: issym
    integer, intent(in) :: irz
    integer, intent(in) :: ii
    integer, intent(in) :: Ntz
    real(rkind), intent(in) :: dBB(1:Ntz, -1:2)
    real(rkind), intent(in) :: XX(1:Ntz)
    real(rkind), intent(in) :: YY(1:Ntz)
    real(rkind), intent(in) :: length(1:Ntz)
    real(rkind), intent(in) :: dRR(1:Ntz, -1:2)
    real(rkind), intent(in) :: dZZ(1:Ntz, -1:2)
    real(rkind), intent(inout) :: dII(1:Ntz)
    real(rkind), intent(inout) :: dLL(1:Ntz)
    real(rkind), intent(inout) :: dPP(1:Ntz)

    logical, intent(in) :: LComputeDerivatives

    integer :: iocons, ideriv, iflag, ifail, idoc
    real(rkind) :: DDl, MMl
    real(rkind) :: constraint(1:Ntz)

    do iocons = 0, 1

      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis): no constraints
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     : no constraints

      ! dBBdmp CONSTRUCTION
      ! ===================

      ! Evaluate derivatives of B^2 w.r.t mu and pflux
      ! ----------------------------------------------
      if (Lconstraint == 1 .or. Lconstraint == 3) then ! first, determine how B^2 varies with mu and dpflux

        iflag = 1
        do ideriv = 1, 2
          !call evaluate_Bsquare(iocons, lvol, dBB, dAt, dAz, XX, YY, length, DDl, MMl, ideriv)! In a subroutine: called
          !somewhere else when semi global constraint
          call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)
        end do

        call tfft(dBB(1:Ntz, 1), dBB(1:Ntz, 2), & ! derivatives of B^2 wrt mu and dpflux
            mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), ifail)
        idoc = 0
        dBBdmp(idoc + 1:idoc + mn, lvol, iocons, 1) = efmn(1:mn) * BBweight(1:mn)! pressure
        dBBdmp(idoc + 1:idoc + mn, lvol, iocons, 2) = cfmn(1:mn) * BBweight(1:mn)! pressure
        idoc = idoc + mn ! even

        ! Add spectral constraints: spectral constraints do not depend on mu or dpflux: thus add nothing
        ! Commented - kept for understanding
        !if( Igeometry.ge.3 ) then
        !    idoc = idoc + mn-1 ! oddd
        !endif

        if (.not.enforce_stell_sym) then
          dBBdmp(idoc + 1:idoc + mn - 1, lvol, iocons, 1) = ofmn(2:mn) * BBweight(2:mn)! pressure
          dBBdmp(idoc + 1:idoc + mn - 1, lvol, iocons, 2) = sfmn(2:mn) * BBweight(2:mn)! pressure
          idoc = idoc + mn - 1 ! oddd

          ! Add spectral constraints: spectral constraints do not depend on mu or dpflux: thus add nothing
          ! Commented - kept for understanding
          !if( Igeometry.ge.3 ) then
          !    idoc = idoc + mn   ! even
          !endif

        end if

      end if

      ! Evaluate B^2 (no derivatives)
      ! -----------------------------------
      ideriv = 0
      iflag = 1

      call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)

      ! dFFdRZ CONSTRUCTION
      ! ===================

      ! B square contribution
      ! ---------------------
      ideriv = -1
      iflag = 0

      call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)

      ! Add derivatives of pressure as well
      call fatal(vvolume(lvol) < small, __FILE__, "shall divide by vvolume(lvol)**(gamma+1.0_rkind) ")

      ! Derivatives of force wrt geometry: In real space.
      ijreal(1:Ntz) = -adiabatic(lvol) * pscale * gamma * dvolume / vvolume(lvol) **  (gamma + 1.0_rkind) + dBB(1:Ntz, -1)

      ! Spectral condensation contribution
      ! ----------------------------------

      dLL(1:Ntz) = 0.0_rkind ! either no spectral constraint, or not the appropriate interface
      dPP(1:Ntz) = 0.0_rkind ! either no spectral constraint, or not the appropriate interface

      if (Igeometry >= 3) then ! spectral constraints are only required in toroidal or extended-cylindrical geometry

        if (innout == 1 .and. iocons == 1) then ! include derivatives of spectral constraints
          if (issym == 0) then ! take derivatives wrt Rbc and Zbs
            if (irz == 0) then ! take derivative wrt Rbc
              dII(1:Ntz) = -im(ii) * sini(1:Ntz, ii) * (XX(1:Ntz) - MMl * iRij(1:Ntz, lvol)) - 2.0_rkind * (mmpp(ii) -&
                  & MMl) * iRbc(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * iZij(1:Ntz, lvol))&
                  & / DDl + Rij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * cosi(1:Ntz, ii)
            else ! take derivative wrt Zbs
              dII(1:Ntz) = +im(ii) * cosi(1:Ntz, ii) * (YY(1:Ntz) - MMl * iZij(1:Ntz, lvol)) - 2.0_rkind * (mmpp(ii) -&
                  & MMl) * iZbs(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * iZij(1:Ntz, lvol))&
                  & / DDl + Zij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * sini(1:Ntz, ii)
            end if
          else ! take derivatives wrt Rbs and Zbc
            if (irz == 0) then
              dII(1:Ntz) = +im(ii) * cosi(1:Ntz, ii) * (XX(1:Ntz) - MMl * iRij(1:Ntz, lvol)) - 2.0_rkind * (mmpp(ii) -&
                  & MMl) * iRbs(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * iZij(1:Ntz, lvol))&
                  & / DDl + Rij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * sini(1:Ntz, ii)
            else
              dII(1:Ntz) = -im(ii) * sini(1:Ntz, ii) * (YY(1:Ntz) - MMl * iZij(1:Ntz, lvol)) - 2.0_rkind * (mmpp(ii) -&
                  & MMl) * iZbc(ii, lvol) * (Rij(1:Ntz, 2, 0) * iRij(1:Ntz, lvol) + Zij(1:Ntz, 2, 0) * iZij(1:Ntz, lvol))&
                  & / DDl + Zij(1:Ntz, 2, 0) * (mmpp(ii) - MMl) * cosi(1:Ntz, ii)
            end if !matches ( irz.eq.0 )
          end if ! matches ( issym.eq.0 )

        else

          dII(1:Ntz) = 0.0_rkind ! either no spectral constraint, or not the appropriate interface

        end if

        constraint(1:Ntz) = + (dRij(1:Ntz, lvol) * tRij(1:Ntz, lvol - 1 + iocons) + dZij(1:Ntz, lvol) * tZij(1:Ntz, lvol -&
            & 1 + iocons)) / length(1:Ntz)

        if (iocons == 0) then ! take derivatives of constraints at inner boundary

          if (innout == 0) then ! derivative wrt inner boundary coefficient
            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tRij(1:Ntz, lvol - 1) - dRij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) /&
                    & length(1:Ntz) + constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) /&
                    & length(1:Ntz)
              else
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tZij(1:Ntz, lvol - 1) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) /&
                    & length(1:Ntz) + constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) /&
                    & length(1:Ntz)
              end if
            else ! if issym.eq.1 : take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tRij(1:Ntz, lvol - 1) + dRij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) /&
                    & length(1:Ntz) + constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) /&
                    & length(1:Ntz)
              else
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tZij(1:Ntz, lvol - 1) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) /&
                    & length(1:Ntz) + constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) /&
                    & length(1:Ntz)
              end if
            end if
          else

            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tRij(1:Ntz, lvol - 1)) / length(1:Ntz) - constraint(1:Ntz) * dRij(1:Ntz&
                    &, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tZij(1:Ntz, lvol - 1)) / length(1:Ntz) - constraint(1:Ntz) * dZij(1:Ntz&
                    &, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            else ! if issym.eq.1 : take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tRij(1:Ntz, lvol - 1)) / length(1:Ntz) - constraint(1:Ntz) * dRij(1:Ntz&
                    &, lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tZij(1:Ntz, lvol - 1)) / length(1:Ntz) - constraint(1:Ntz) * dZij(1:Ntz&
                    &, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            end if
          end if

        else

          if (innout == 0) then ! derivative wrt inner boundary coefficient

            if (issym == 0) then ! take derivatives wrt Rbc and Zbs
              if (irz == 0) then
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tRij(1:Ntz, lvol)) / length(1:Ntz) + constraint(1:Ntz) * dRij(1:Ntz,&
                    & lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tZij(1:Ntz, lvol)) / length(1:Ntz) + constraint(1:Ntz) * dZij(1:Ntz,&
                    & lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            else ! take derivatives wrt Rbs and Zbc
              if (irz == 0) then
                dLL(1:Ntz) = + (-sini(1:Ntz, ii) * tRij(1:Ntz, lvol)) / length(1:Ntz) + constraint(1:Ntz) * dRij(1:Ntz,&
                    & lvol) * sini(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              else
                dLL(1:Ntz) = + (-cosi(1:Ntz, ii) * tZij(1:Ntz, lvol)) / length(1:Ntz) + constraint(1:Ntz) * dZij(1:Ntz,&
                    & lvol) * cosi(1:Ntz, ii) / length(1:Ntz) / length(1:Ntz)
              end if
            end if
          else

            if (Igeometry == 3 .and. lvol == 1) then ! need to accomodate derivatives of coordinate axis

              if (issym == 0) then ! take derivatives wrt Rbc and Zbs
                if (irz == 0) then
                  dLL(1:Ntz) = ( & ! d/dRbc
                      + (cosi(1:Ntz, ii) - dRodR(1:Ntz, 0, ii)) * tRij(1:Ntz, lvol) - dRij(1:Ntz, lvol) * im(ii) *&
                      & sini(1:Ntz, ii) + (-dZodR(1:Ntz, 0, ii)) * tZij(1:Ntz, lvol) - constraint(1:Ntz) * (dRij(1:Ntz,&
                      & lvol) * (cosi(1:Ntz, ii) - dRodR(1:Ntz, 0, ii)) + dZij(1:Ntz, lvol) * (-dZodR(1:Ntz, 0, ii)) ) /&
                      & length(1:Ntz)) / length(1:Ntz)
                else
                  dLL(1:Ntz) = ( & ! d/dZbs
                      + (-dRodZ(1:Ntz, 1, ii)) * tRij(1:Ntz, lvol) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii) +&
                      & (sini(1:Ntz, ii) - dZodZ(1:Ntz, 1, ii)) * tZij(1:Ntz, lvol) - constraint(1:Ntz) * (dRij(1:Ntz,&
                      & lvol) * (-dRodZ(1:Ntz, 1, ii)) + dZij(1:Ntz, lvol) * (sini(1:Ntz, ii) - dZodZ(1:Ntz, 1, ii)) ) /&
                      & length(1:Ntz)) / length(1:Ntz)
                end if
              else
                if (irz == 0) then
                  dLL(1:Ntz) = ( & ! d/dRbs
                      + (sini(1:Ntz, ii) - dRodR(1:Ntz, 1, ii)) * tRij(1:Ntz, lvol) + dRij(1:Ntz, lvol) * im(ii) *&
                      & cosi(1:Ntz, ii) + (-dZodR(1:Ntz, 1, ii)) * tZij(1:Ntz, lvol) - constraint(1:Ntz) * (dRij(1:Ntz,&
                      & lvol) * (sini(1:Ntz, ii) - dRodR(1:Ntz, 1, ii)) + dZij(1:Ntz, lvol) * (-dZodR(1:Ntz, 1, ii)) ) /&
                      & length(1:Ntz)) / length(1:Ntz)
                else
                  dLL(1:Ntz) = ( & ! d/dZbs
                      + (-dRodZ(1:Ntz, 0, ii)) * tRij(1:Ntz, lvol) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii) +&
                      & (cosi(1:Ntz, ii) - dZodZ(1:Ntz, 0, ii)) * tZij(1:Ntz, lvol) - constraint(1:Ntz) * (dRij(1:Ntz,&
                      & lvol) * (-dRodZ(1:Ntz, 0, ii)) + dZij(1:Ntz, lvol) * (cosi(1:Ntz, ii) - dZodZ(1:Ntz, 0, ii)) ) /&
                      & length(1:Ntz)) / length(1:Ntz)
                end if
              end if
            else
              if (issym == 0) then ! take derivatives wrt Rbc and Zbs
                if (irz == 0) then
                  dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tRij(1:Ntz, lvol) - dRij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) /&
                      & length(1:Ntz) - constraint(1:Ntz) * dRij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) /&
                      & length(1:Ntz)
                else
                  dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tZij(1:Ntz, lvol) + dZij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) /&
                      & length(1:Ntz) - constraint(1:Ntz) * dZij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) /&
                      & length(1:Ntz)
                end if
              else ! take derivatives wrt Rbs and Zbc
                if (irz == 0) then
                  dLL(1:Ntz) = + (+sini(1:Ntz, ii) * tRij(1:Ntz, lvol) + dRij(1:Ntz, lvol) * im(ii) * cosi(1:Ntz, ii)) /&
                      & length(1:Ntz) - constraint(1:Ntz) * dRij(1:Ntz, lvol) * sini(1:Ntz, ii) / length(1:Ntz) /&
                      & length(1:Ntz)
                else
                  dLL(1:Ntz) = + (+cosi(1:Ntz, ii) * tZij(1:Ntz, lvol) - dZij(1:Ntz, lvol) * im(ii) * sini(1:Ntz, ii)) /&
                      & length(1:Ntz) - constraint(1:Ntz) * dZij(1:Ntz, lvol) * cosi(1:Ntz, ii) / length(1:Ntz) /&
                      & length(1:Ntz)
                end if
              end if
            end if
          end if

        end if

      end if

      ! Map to Fourier space
      call tfft(ijreal(1:Ntz), dII(1:Ntz), & ! recall that ijreal contains derivatives of pressure term
          mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), ifail)

      call tfft(dPP(1:Ntz), dLL(1:Ntz), mn, im(1:mn), in(1:mn), evmn(1:mn), odmn(1:mn), comn(1:mn), simn(1:mn),&
          & ifail)! evmn and odmn are available as workspace

      call fatal(lvol - 1 + innout > Mvol, __FILE__, "psifactor needs attention ")

      idoc = 0

      ! Plasma and magnetic pressure
      dFFdRZ(idoc + 1:idoc + mn, iocons, idof, innout, lvol) = +efmn(1:mn) * psifactor(ii, lvol - 1 + innout) *&
          & BBweight(1:mn)

      idoc = idoc + mn ! even
      if (Igeometry >= 3) then ! Add spectral constraints
        ! spectral condensation + poloidal length constraint
        dFFdRZ(idoc + 1:idoc + mn - 1, iocons, idof, innout, lvol) = -sfmn(2:mn) * psifactor(ii, lvol - 1 + innout) *&
            & epsilon - simn(2:mn) * psifactor(ii, lvol - 1 + innout) * sweight(lvol)
        idoc = idoc + mn - 1 ! odd
      end if

      if (.not.enforce_stell_sym) then ! Construct non-stellarator symmetric terms

        ! Plasma and magnetic pressure
        dFFdRZ(idoc + 1:idoc + mn - 1, iocons, idof, innout, lvol) = +ofmn(2:mn) * psifactor(ii, lvol - 1 + innout) *&
            & BBweight(2:mn)

        idoc = idoc + mn - 1 ! odd
        if (Igeometry >= 3) then ! Add spectral constraints
          ! spectral condensation + poloidal length constraint
          dFFdRZ(idoc + 1:idoc + mn, iocons, idof, innout, lvol) = -cfmn(1:mn) * psifactor(ii, lvol - 1 + innout) *&
              & epsilon - comn(1:mn) * psifactor(ii, lvol - 1 + innout) * sweight(lvol)
          idoc = idoc + mn ! even
        end if

      end if
    end do

  end subroutine evaluate_dBB

  subroutine lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)
    integer, intent(in) :: lvol, iocons, ideriv, Ntz, iflag
    real(rkind) :: dAt(1:Ntz, -1:2), dAz(1:Ntz, -1:2), XX(1:Ntz), YY(1:Ntz), DDl, MMl

    real(rkind) :: IIl(1:Ntz), length(1:Ntz), dLL(1:Ntz)
    integer :: Lcurvature, ifail, ivol, id
    real(rkind) :: dBB(1:Ntz, -1:2), lss

    dAt(1:Ntz, -1:2) = 0.0_rkind ! initialize intent out
    dAz(1:Ntz, -1:2) = 0.0_rkind ! initialize intent out

    lss = 2.0_rkind * iocons - 1.0_rkind ! recall that iocons is effective local radial coordinate

    Lcurvature = 1

    call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)! get coordinates and derivatives wrt Rj, Zj, at specific radial location

    call build_vector_potential(lvol, iocons, 0)

    ! Map to real space
    call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:Ntz, 0), dAz(1:Ntz, 0))

    id = ideriv
    if (id == 0) then

      dBB(1:Ntz, id) = 1.0_rkind / 2.0_rkind * (dAz(1:Ntz, id) * dAz(1:Ntz, id) * guvij(1:Ntz, 2, 2, id) - 2.0_rkind *&
          & dAz(1:Ntz, id) * dAt(1:Ntz, id) * guvij(1:Ntz, 2, 3, id) + dAt(1:Ntz, id) * dAt(1:Ntz, id) * guvij(1:Ntz, 3,&
          & 3, id)) * inv_jac(1:Ntz) ** 2

    else
      ! Compute covariant vector potential. Stored in efmn, ofmn, cfmn, sfmn.
      call build_vector_potential(lvol, iocons, id)
      ! Map to real space
      call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:Ntz, id), dAz(1:Ntz, id))

      dBB(1:Ntz, id) = 1.0_rkind / 2.0_rkind * (dAz(1:Ntz, id) * dAz(1:Ntz, 0) * guvij(1:Ntz, 2, 2, 0) - 2.0_rkind *&
          & dAz(1:Ntz, id) * dAt(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 0) + dAt(1:Ntz, id) * dAt(1:Ntz, 0) * guvij(1:Ntz, 3, 3,&
          & 0) + dAz(1:Ntz, 0) * dAz(1:Ntz, id) * guvij(1:Ntz, 2, 2, 0) - 2.0_rkind * dAz(1:Ntz, 0) * dAt(1:Ntz, id) *&
          & guvij(1:Ntz, 2, 3, 0) + dAt(1:Ntz, 0) * dAt(1:Ntz, id) * guvij(1:Ntz, 3, 3, 0)) * inv_jac(1:Ntz) ** 2
    end if

    ! If derivatives w.r.t geometry, take into account metric derivatives
    if (ideriv == -1) then
      ! Get coordinate metrics and their derivatives wrt Rj, Zj on interface
      lss = 2.0_rkind * iocons - 1.0_rkind
      Lcurvature = 4
      call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)

      dBB(1:Ntz, id) = dBB(1:Ntz, id) + 1.0_rkind / 2.0_rkind * (dAz(1:Ntz, 0) * dAz(1:Ntz, 0) * guvij(1:Ntz, 2, 2, 1) -&
          2.0_rkind * dAz(1:Ntz, 0) * dAt(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 1) + dAt(1:Ntz, 0) * dAt(1:Ntz, 0) *&
          guvij(1:Ntz, 3, 3, 1)) * inv_jac(1:Ntz) ** 2 - dBB(1:Ntz, 0) * 2.0_rkind * sg(1:Ntz, 1) * inv_jac(1:Ntz)
    end if

    ijreal(1:Ntz) = adiabatic(lvol) * pscale / vvolume(lvol) ** gamma + dBB(1:Ntz, 0)! p + B^2/2

    if (iflag /= 1) then
      ! compute spectral constraints
      select case (Igeometry)
      case (1:2)
        dLL(1:Ntz) = 0.0_rkind ! placeholder
        IIl(1:Ntz) = 0.0_rkind ! placeholder

      case (3)
        do ivol = 0, 1
          call invfft(mn, im(1:mn), in(1:mn), iRbc(1:mn, lvol - 1 + ivol), iRbs(1:mn, lvol - 1 + ivol), iZbc(1:mn, lvol -&
              & 1 + ivol), iZbs(1:mn, lvol - 1 + ivol), iRij(1:Ntz, lvol - 1 + ivol), iZij(1:Ntz, lvol - 1 +&
              & ivol))

          call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * iRbs(1:mn, lvol - 1 + ivol), -im(1:mn) * iRbc(1:mn, lvol - 1 +&
              & ivol), im(1:mn) * iZbs(1:mn, lvol - 1 + ivol), -im(1:mn) * iZbc(1:mn, lvol - 1 + ivol), &
              & tRij(1:Ntz, lvol - 1 + ivol), tZij(1:Ntz, lvol - 1 + ivol))
        end do

        dRij(1:Ntz, lvol) = iRij(1:Ntz, lvol) - iRij(1:Ntz, lvol - 1)
        dZij(1:Ntz, lvol) = iZij(1:Ntz, lvol) - iZij(1:Ntz, lvol - 1)

        length(1:Ntz) = sqrt(dRij(1:Ntz, lvol) ** 2 + dZij(1:Ntz, lvol) ** 2)

        dLL(1:Ntz) = (dRij(1:Ntz, lvol) * tRij(1:Ntz, lvol - 1 + iocons) + dZij(1:Ntz, lvol) * tZij(1:Ntz, lvol - 1 +&
            iocons)) / length(1:Ntz)

        if (iocons == 1) then ! include spectral condensation constraints: local to interface, i.e. no tri-diagonal
          ! structure
          efmn(1:mn) = (mmpp(1:mn)) * iRbc(1:mn, lvol)
          sfmn(1:mn) = (mmpp(1:mn)) * iZbs(1:mn, lvol)
          if (.not.enforce_stell_sym) then
            ofmn(1:mn) = (mmpp(1:mn)) * iRbs(1:mn, lvol)
            cfmn(1:mn) = (mmpp(1:mn)) * iZbc(1:mn, lvol)
          else
            ofmn(1:mn) = 0.0_rkind
            cfmn(1:mn) = 0.0_rkind
          end if

          call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), XX(1:Ntz),&
              & YY(1:Ntz))

          if (enforce_stell_sym) then
            DDl = sum((iRbc(1:mn, lvol) ** 2 + iZbs(1:mn, lvol) ** 2))
            MMl = sum(mmpp(1:mn) * (iRbc(1:mn, lvol) ** 2 + iZbs(1:mn, lvol) ** 2) ) / DDl
          else
            DDl = sum((iRbc(1:mn, lvol) ** 2 + iZbs(1:mn, lvol) ** 2 + iRbs(1:mn, lvol) ** 2 + iZbc(1:mn, lvol) ** 2))
            MMl = sum(mmpp(1:mn) * (iRbc(1:mn, lvol) ** 2 + iZbs(1:mn, lvol) ** 2 + iRbs(1:mn, lvol) ** 2 + iZbc(1:mn, lvol)&
                &** 2) ) / DDl
          end if

          IIl(1:Ntz) = tRij(1:Ntz, lvol) * (XX(1:Ntz) - MMl * iRij(1:Ntz, lvol)) + tZij(1:Ntz, lvol) * (YY(1:Ntz) - MMl *&
              & iZij(1:Ntz, lvol))

        else ! matches if( iocons.eq.1 )
          IIl(1:Ntz) = 0.0_rkind ! placeholder
        end if
      end select

      ifail = 0
      call tfft(ijreal(1:Ntz), IIl(1:Ntz), & ! compute force-imbalance and spectral constraints
          mn, im(1:mn), in(1:mn), Bemn(1:mn, lvol, iocons), Bomn(1:mn, lvol, iocons), Iemn(1:mn, lvol), Iomn(1:mn, lvol),&
          & ifail)

      if (Igeometry >= 3) then ! add minimal length constraint

        ifail = 0
        ijimag(1:Ntz) = 0.0_rkind

        call tfft(dLL(1:Ntz), ijimag(1:Ntz), mn, im(1:mn), in(1:mn), Semn(1:mn, lvol, iocons), Somn(1:mn, lvol,&
            & iocons), Pemn(1:mn, lvol, iocons), Pomn(1:mn, lvol, iocons), ifail)
      end if
    end if
  end subroutine lforce

  subroutine hessian3D_dFFdRZ(lvol, idof, innout, issym, irz, ii, dBB, XX, YY, length, dRR, dZZ, dII,&
        & dLL, dPP, Ntz)

    ! Evaluate the derivative of the square of the magnetic field modulus.

    integer :: iocons, lvol, ideriv, iflag, innout, issym, irz, ii, ifail, idoc, idof, Ntz
    real(rkind) :: DDl, MMl
    real(rkind) :: XX(1:Ntz), YY(1:Ntz), dBB(1:Ntz, -1:2), dII(1:Ntz), dLL(1:Ntz)
    real(rkind) :: dPP(1:Ntz), length(1:Ntz), dRR(1:Ntz, -1:2), dZZ(1:Ntz, -1:2)
    real(rkind) :: ddFcol1(1:Ntz), ddFcol2(1:Ntz), ddFcol3(1:Ntz), ddFcol4(1:Ntz)
    
    do iocons = 0, 1

      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis); no constraints;
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; no constraints;

      ! Evaluate B^2 (no derivatives)
      ideriv = 0
      iflag = 0
      
      call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)

      ! hessian_dFFdRZ CONSTRUCTION

      ! B square contribution
      ideriv = -1
      iflag = 1

      call lforce(lvol, iocons, ideriv, Ntz, dBB, XX, YY, length, DDl, MMl, iflag)

      ! Add derivatives of pressure as well
      call fatal(vvolume(lvol) < small, __FILE__, "shall divide by vvolume(lvol)**(gamma+1.0_rkind) ")

      ! Derivatives of force wrt geometry; In real space.
      ijreal(1:Ntz) = -adiabatic(lvol) * pscale * gamma * dvolume &
          / vvolume(lvol)**(gamma + 1.0_rkind) + dBB(1:Ntz, -1)

      if (enforce_stell_sym) then
        if (irz == 0) then !derivative wrt Rbs of d/dr(dF)

            if (innout == iocons) then
            ddFcol1(1:Ntz) = ijreal(1:Ntz) * Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0) & !Frr       d/dr(dw/dr)
                + (dBB(1:Ntz, 0) + adiabatic(lvol) * pscale / vvolume(lvol)**gamma) * Zij(1:Ntz, 2, 0&
                ) * cosi(1:Ntz, ii)

            ddFcol2(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) & !Fzr        d/dr(dw/dz)
                + (dBB(1:Ntz, 0) + adiabatic(lvol) * pscale / vvolume(lvol)**gamma) * Rij(1:Ntz, 2, 0&
                ) * cosi(1:Ntz, ii) &
                + (dBB(1:Ntz, 0) + adiabatic(lvol) * pscale / vvolume(lvol)**gamma) * Rij(1:Ntz, 0, 0&
                ) * sini(1:Ntz, ii) * (-im(ii))
            else

            ddFcol1(1:Ntz) = ijreal(1:Ntz) * Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0) !Frr       d/dr(dw/dr)
            ddFcol2(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) !Fzr        d/dr(dw/dz)
            end if
        else !derivative wrt to Zbs of d/dz(dF)

            if (innout == iocons) then
            ddFcol3(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Zij(1:Ntz, 2, 0) & !Frz      d/dz(dw/dr)
                + (dBB(1:Ntz, 0) + adiabatic(lvol) * pscale / vvolume(lvol)**gamma) * Rij(1:Ntz, 0, 0&
                ) * im(ii) * cosi(1:Ntz, ii)
            ddFcol4(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) !Fzz    d/dz(dw/dz)
            else
            ddFcol3(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Zij(1:Ntz, 2, 0) !Frz      d/dz(dw/dr)
            ddFcol4(1:Ntz) = ijreal(1:Ntz) * Rij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) !Fzz    d/dz(dw/dz)
            end if
        end if
      else
        ddFcol1(1:Ntz) = 0.0_rkind
        ddFcol2(1:Ntz) = 0.0_rkind
        ddFcol3(1:Ntz) = 0.0_rkind
        ddFcol4(1:Ntz) = 0.0_rkind
      end if

      !Map to Fourier space

      call tfft(ddFcol1(1:Ntz), ddFcol2(1:Ntz), & ! recall that ijreal contains derivatives of pressure term;
          mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), ifail)

      call tfft(ddFcol3(1:Ntz), ddFcol4(1:Ntz), &
          mn, im(1:mn), in(1:mn), evmn(1:mn), odmn(1:mn), comn(1:mn), simn(1:mn), ifail) ! evmn and odmn are available as workspace;

      if (irz == 0 .and. ii == 1) then
          idoc = 0
          denergydrr(idoc + 1:idoc + mn, lvol, iocons, idof, innout) = -efmn(1:mn) * 1.0_rkind / 2.0_rkind !1 wrr
          idoc = idoc + mn
          denergydrr(idoc + 1:idoc + mn - 1, lvol, iocons, idof, innout) = -sfmn(2:mn) * 1.0_rkind &
                  / 2.0_rkind ! 2 wrz
          idoc = idoc + mn - 1
      end if
      
      if (irz == 0 .and. ii > 1) then
          idoc = 0
          denergydrr(idoc + 1:idoc + mn, lvol, iocons, idof, innout) = -efmn(1:mn) !1 wrr
          idoc = idoc + mn
          denergydrr(idoc + 1:idoc + mn - 1, lvol, iocons, idof, innout) = -sfmn(2:mn) ! 2 wrz
          idoc = idoc + mn - 1
      end if

      if (irz == 1 .and. ii > 1) then
          idoc = 0
          denergydzr(idoc + 1:idoc + mn - 1, lvol, iocons, idof, innout) = evmn(1:mn - 1) !wzr
          idoc = idoc + mn
          denergydzr(idoc + 1:idoc + mn - 1, lvol, iocons, idof, innout) = simn(2:mn) !wzz
          idoc = idoc + mn - 1
      end if

      ! Spectral condensation contribution
      dLL(1:Ntz) = 0.0_rkind !  no spectral constraint, or not the appropriate interface;
      dPP(1:Ntz) = 0.0_rkind !  no spectral constraint, or not the appropriate interface;
      dII(1:Ntz) = 0.0_rkind !  no angle/spectral width constraint
    end do ! end of do iocons;

  end subroutine hessian3D_dFFdRZ

end module forces_mod
