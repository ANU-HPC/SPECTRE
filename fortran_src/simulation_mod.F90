module simulation_mod
#include "mpi_precomp.h"
  USE_MPI

  use allglobal_mod, only: cpus, myid, ext, get_hidden, MPI_COMM_SPEC, write_spec_namelist, ncpu, Mvol, Ntz, mn, &
      beltramierror, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion, Btemn, Bzemn, Btomn, Bzomn, efmn, ofmn, cfmn, sfmn, &
      IPDt, ImagneticOK, dtflux, Iquad, lmns, Nt, Nz, diotadxup, IsMyVolume, IsMyVolumeValue, WhichCpuID, dlambdaout, diotadxup, &
      set_mpi_comm, wrtend, read_inputlists_from_file, check_inputs, broadcast_inputs, skip_write, im, in, LGdof, NGdof, iRbc, &
      iZbs, iRbs, iZbc, BBe, IIo, BBo, IIe, vvolume, dpflux, ForceErr, BnsErr, iBns, iBnc, iVns, iVnc, nfreeboundaryiterations, &
      first_free_bound
  use constants_mod, only: version, pi2
  use cputiming_mod
  use diagnostics_mod, only: compute_beltrami_error
  use errors_mod, only: fatal
  use forces_mod, only: dforce
  use hessian_mod, only: compute_hessian
  use input_list_mod, only: initialize_inputs, Wxspech, nPtrj, nPpts, Igeometry, Lcheck, Nvol, odetol, Isurf, Ivolume, mu, &
      Wmacros, Ltransform, Lsvdiota, Nfp, Lrad, tflux, pflux, phiedge, pressure, pscale, helicity, Ladiabatic, adiabatic, &
      gamma, Rbc, Zbs, Rbs, Zbc, Lconstraint, Lfreebound, Lbnszero, mfreeits, gBntol, gBnbld, vcasingtol, LautoinitBn, &
      LHevalues, LHevectors, LHmatrix, Lzerovac, enforce_stell_sym, Wreadin
  use io_mod, only: ounit, lunit
  use magnetic_field_mod, only: compute_rotational_transform
  use freeb_mod, only: compute_bnorm_plasmab
  use memory_allocation_mod, only: sallocate
  use mpi_collectives_mod, only: broadcast
  use numerical_mod, only: set_fftw_plan
  use packing_mod, only: packxi
  use plasma_current_mod, only: lbpol
  use poincare_mod, only: construct_poincare_plot
  use precisions_mod, only: rkind, small, vsmall, machprec
  use preset_mod, only: initialize_internal_arrays
  use profiles_distribution_mod, only: broadcast_profile_data
  use sphdf5_mod, only: init_outfile, mirror_input_to_outfile, init_convergence_output, hdfint, finish_outfile, write_grid, write_stability
  use vector_potential_writer_mod, only: write_vector_potentials
  use volume_mod, only: volume, lregion
  use field_mod, only: alloc_ntz_arrays

  implicit none

  private
  public :: xspech, spectre, final_diagnostics, ending

contains

  subroutine xspech
    integer :: ierr
    real(rkind) :: cput, cpuo = 0.0_rkind

    character :: ldate * 8, ltime * 10

    call MPI_INIT(ierr)

    ! set default communicator to MPI_COMM_WORLD
#ifdef NO_MPI_F08
    call set_mpi_comm(MPI_COMM_WORLD)
#else
    call set_mpi_comm(MPI_COMM_WORLD%mpi_val)
#endif

    ! set initial time
    cpus = MPI_Wtime()
    cpuo = cpus

    ! explicitly enable writing of HDF5 output file
    skip_write = .false.

    ! print header: version of SPECTRE, compilation info, current date and time, machine precision
    cput = MPI_Wtime()
    if (myid == 0) then

      ! screen output header
      write(ounit, '("xspech : ", 10x ," : version = "F5.2)')version

      call date_and_time(ldate, ltime)
      write(ounit, '("xspech : ", 10x ," : ")')
      write(ounit, 1000)cput - cpus, ldate(1:4), ldate(5:6), ldate(7:8), ltime(1:2), ltime(3:4), ltime(5:6), machprec,&
          & vsmall, small

      write(ounit, '("xspech : ", 10x ," : ")')
      write(ounit, '("xspech : ",f10.2," : parallelism : ncpu=",i3)')cput - cpus, ncpu

      ! read command-line arguments
      call read_command_args()

      ! initialize input arrays into a default state
      call initialize_inputs()

      write(ounit, '("xspech : ", 10x ," : ")')
      write(ounit, '("xspech : ",f10.2," : begin execution calling global:readin ")')cput - cpus

      call read_inputlists_from_file()

      ! check that data from input file is within allowed ranges etc.
      call check_inputs()

    end if

    ! broadcast input file contents
    call broadcast_inputs()

    ! initialize internal arrays based on data from input file
    call initialize_internal_arrays()

    ! initialize HDF5 library and open output file ext.h5 for writing during execution
    call init_outfile()

    ! mirror input file contents to output file
    call mirror_input_to_outfile()

    if (myid == 0) then ! save restart file
      call wrtend()! write initial restart file
    end if

    call init_convergence_output()

    ! MAIN SUBROUTINE: iterate until converged or #iterations exceeds limit
    call spectre()

    ! some final diagnostics: compute errors, Poincare plots, ....
    call final_diagnostics()

    ! post-processing: magnetic field evaluated on a grid
    call write_grid()

    if (myid == 0) then

      call wrtend()
    end if

    ! write final outputs to HDF5 file
    call hdfint()

    ! close HDF5 output file
    call finish_outfile()

    ! print ending info
    call ending()

    ! wait for writing to finish
    call MPI_Barrier(MPI_COMM_SPEC, ierr)

    if (myid == 0) then
      cput = MPI_Wtime()
      write(ounit, '("xspech : ", 10x ," :")')
      write(ounit, '("xspech : ",f10.2," : myid=",i3," : time="f8.2"m = "f6.2"h = "f5.2"d ")')cput - cpus, myid, (cput &
          &- cpus) / [60, 60 * 60, 24 * 60 * 60]
    end if

    call mpi_finalize(ierr)

    stop

    1000 format("xspech : ", f10.2, " : date=", a4, "/", a2, "/", a2, " , ", a2, ":", a2, ":", a2, " ; machine precision=", &
        es9.2, " ; vsmall=", es9.2, " ; small=", es9.2, " ")
  end subroutine xspech

  subroutine read_command_args
    integer :: ierr
    real(rkind) :: cput

    integer :: iargc, iarg, numargs, extlen, sppos

    character(len=100) :: arg

    if (myid == 0) then

      cput = MPI_Wtime()

      ! first command-line argument is likely ext or ext.sp
      call getarg(1, arg)

      write(ounit, '("rdcmdl : ", 10x ," : ")')
      select case (trim(arg))
      case ("", "-h", "--help")
        write(ounit, '("rdcmdl : ", 10x ," : file extension must be given as first command line argument ")')
        write(ounit, '("rdcmdl : ", 10x ," : Usage: <mpiexec> xspec input_file <arguments>")')
        write(ounit, '("rdcmdl : ", 10x ," : Other options:")')
        write(ounit, '("rdcmdl : ", 10x ," :     -h / --help :  print help information.")')
        write(ounit, '("rdcmdl : ", 10x ," :     -i / --init :  generate a template input file.")')
        write(ounit, '("rdcmdl : ", 10x ," : Additional arguments:")')
        write(ounit, '("rdcmdl : ", 10x ," :     -readin : print debugging information during reading inputs")')
        call MPI_ABORT(MPI_COMM_SPEC, 0, ierr)
      case ("-i", "--init")
        write(ounit, '("rdcmdl : ", 10x ," : write a template input file in example.sp")')
        call write_spec_namelist()
        call MPI_ABORT(MPI_COMM_SPEC, 0, ierr)
      case default
        extlen = len_trim(arg)
        sppos = index(arg, ".sp", .true.)! search for ".sp" from the back of ext
        if (sppos == extlen - 2) then ! check if ext ends with ".sp"
          arg = arg(1:extlen - 3)! if this is the case, remove ".sp" from end of ext
        end if
        ext = trim(arg)

        write(ounit, '("rdcmdl : ", 10x ," : ")')
        write(ounit, '("rdcmdl : ",f10.2," : ext = ",a100)')cput - cpus, ext
      end select

      numargs = iargc()

      if (numargs > 1) then
        iarg = 1
        do while (iarg < numargs)
          iarg = iarg + 1
          call getarg(iarg, arg)
          select case (arg)
          case ("-readin")
            Wreadin = .true.
          case ("-p4pg")
            iarg = iarg + 1
            call getarg(iarg, arg)! TODO: what is this?
          case ("-p4wd")
            iarg = iarg + 1
            call getarg(iarg, arg)! TODO: what is this?
          case default
            write(ounit, '("rdcmdl : ",f10.2," : myid=",i3," : argument not recognized arg = ",a100)')cput - cpus, myid&
                & , arg
          end select
        end do
      end if

    end if

  end subroutine read_command_args

  subroutine spectre
    real(rkind) :: cput, cpuo

    logical :: LComputeDerivatives, LContinueFreeboundaryIterations, exist, LupdateBn, LComputeAxis
    integer :: imn, lmn, lNfp, lim, lin, ii
    integer :: vvol, vflag
    real(rkind) :: lastcpu, lRwc, lRws, lZwc, lZws, lItor, lGpol, lgBc, lgBs
    real(rkind), allocatable :: position(:), gradient(:)
    character :: pack
    integer :: mfreeits_old
    real(rkind) :: gBnbld_old

    real(rkind) :: ohessian(1:NGdof, 1:NGdof)

    cpuo = MPI_Wtime()

    call fatal(NGdof < 0, __FILE__, "counting error ")

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call sallocate(position, 0, NGdof, 0.0_rkind)! position : NGdof = #geometrical degrees-of-freedom was computed in
    ! initialize_internal_arrays

    nfreeboundaryiterations = -1

    ! This is the free-boundary iteration loop (implemented using GOTO)
    9000 nfreeboundaryiterations = nfreeboundaryiterations + 1

    ! run fix_boundary for the first free_boundary iteration if LautoinitBn was set to 1
    if (Lfreebound == 1 .and. LautoinitBn == 1) then
      if (nfreeboundaryiterations == 0) then ! first iteration
        first_free_bound = .true.
        !Mvol = Nvol
        gBnbld_old = gBnbld
        gBnbld = 0.0_rkind
        mfreeits_old = mfreeits
        mfreeits = 1
        if (myid == 0) write(ounit, '("xspech : ",10X," : First iteration of free boundary calculation : update Bns from&
            &           plasma.")')
      else
        first_free_bound = .false.
        !Mvol = Nvol + Lfreebound
        gBnbld = gBnbld_old
        mfreeits = mfreeits_old
      end if
    end if

    if (NGdof > 0) then ! pack geometry into vector

      pack = 'P'
      LComputeAxis = .true.
      call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol),&
          & iZbc(1:mn, 0:Mvol), pack, .false., LComputeAxis)

    end if

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      vflag = 0
      call volume(vvol, vflag, Nt, Nz)! compute volume

      if (Ladiabatic == 0) adiabatic(vvol) = pressure(vvol) * vvolume(vvol) ** gamma ! initialize adiabatic constants using
      ! supplied pressure profile

    end do

    if (Mvol > Nvol) then
      adiabatic(Mvol) = 0.0_rkind
      pressure(Mvol) = 0.0_rkind ! these are never used
    end if

    if (Wxspech .and. myid == 0) then
      cput = MPI_Wtime()
      write(ounit, '("xspech : ",f10.2," : myid=",i3," adiabatic constants = "999es13.5)')cput - cpus, myid,&
          & adiabatic(1:Mvol)
    end if

    if (NGdof > 0) then
      pack = 'U' ! unpack geometrical degrees of freedom
      LComputeAxis = .true.
      call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol),&
          & iZbc(1:mn, 0:Mvol), pack, .false., LComputeAxis)
    end if

    ! set/reset input variables

    ! if( Lconstraint.lt.2 ) helicity(1:Nvol) = lABintegral(1:Nvol) ! updated ``input'' quantity
    pressure(1:Mvol) = adiabatic(1:Mvol) / vvolume(1:Mvol) ** gamma ! this matches construction of adiabatic above

    call sallocate(gradient, 0, NGdof, 0.0_rkind)

    lastcpu = MPI_Wtime()

    LComputeDerivatives = .false.
    LComputeAxis = .true.

    call dforce(NGdof, position(0:NGdof), gradient(0:NGdof), LComputeDerivatives, LComputeAxis)! (re-)calculate Beltrami
    ! fields

    deallocate(gradient)

    if (myid == 0) then
      cput = MPI_Wtime()
      write(ounit, 1000)cput - cpus, nfreeboundaryiterations, ForceErr, cput - lastcpu, "|BB|e", log10(BBe(1:min(Mvol -&
          & 1, 28)))
      if (Igeometry >= 3) then ! include spectral constraints
        write(ounit, 1001)"|II|o", log10(IIo(1:min(Mvol - 1, 28)))
      end if
      if (.not.enforce_stell_sym) then
        write(ounit, 1001)"|BB|o", log10(BBo(1:min(Mvol - 1, 28)))
        if (Igeometry >= 3) then ! include spectral constraints
          write(ounit, 1001)"|II|e", log10(IIe(1:min(Mvol - 1, 28)))
        end if
      end if
    end if

    1000 format("xspech : ", f10.2, " : #freeits=", i3, " ; ", :, "|f|=", es12.5, " ; ", :, "time=", f10.2, "s", :, " log", a5, &
        ":=", 28f6.2," ...")
    1001 format("xspech : ", 10x, " :          ", 3x, " ; ", :, "    ", 12x, "   ", :, "     ", 10x, "  ", :, " log", a5, ":=", &
        28f6.2, " ...")

    if (Lconstraint == 2 .and. (Lcheck == 5 .or. LHevalues .or. LHevectors .or. LHmatrix)) then

      if (myid == 0) then
        cput = MPI_Wtime()
        write(ounit, '("xspech : ", 10x ," : -------------------Stability Evaluations------------------ ")')
        write(ounit, '("xspech : ",f10.2," : myid=",i3,"  calling hessian see .ext.hessian.myid ")')cput - cpus, myid
      end if

      call compute_hessian(position(0:NGdof), ohessian)

      call write_stability(ohessian, ohessian(1:NGdof, 1)*0.0_rkind, ohessian(1:NGdof, 1)*0.0_rkind, ohessian*0.0_rkind, NGdof)
    end if

    select case (Igeometry)
    case (1)
      tflux(1) = dtflux(1)
      pflux(1) = dpflux(1)
    case (2:3)
      tflux(1) = dtflux(1)
      pflux(1) = 0.0_rkind
    end select

    do vvol = 2, Mvol
      tflux(vvol) = tflux(vvol - 1) + dtflux(vvol)
      pflux(vvol) = pflux(vvol - 1) + dpflux(vvol)
    end do

    tflux(1:Mvol) = tflux(1:Mvol) * pi2 / phiedge ! this is the "inverse" operation defined in initialize_internal_arrays
    pflux(1:Mvol) = pflux(1:Mvol) * pi2 / phiedge

    LContinueFreeboundaryIterations = .false.

    LupdateBn = .false. ! default

    if (Lfreebound == 1) then
      if (mfreeits > 0 .and. nfreeboundaryiterations < mfreeits) LupdateBn = .true.
      if (mfreeits < 0) LupdateBn = .true.
    end if

    if (Lbnszero) then
      LupdateBn = .false.
      First_free_bound = .false.
      iBns(1:mn) = 0.0_rkind
      if (.not.enforce_stell_sym) iBnc(1:mn) = 0.0_rkind
      if ( myid == 0 ) write(ounit, *)"Bns is set to 0. No picard iteration..."
    end if

    if (LupdateBn) then

      Mvol = Nvol + Lfreebound

      lastcpu = MPI_Wtime()

      call compute_bnorm_plasmab(mn, Ntz, efmn(1:mn), ofmn(1:mn))! compute normal field etc. on computational boundary

      if (mn == 1) then
        if (enforce_stell_sym) bnserr = 0.0_rkind !TODO: NOT SURE, this should test if bns is actually 0
        if (.not.enforce_stell_sym) bnserr = sum(abs(iBnc(1:mn) - efmn(1:mn))) / (mn)
      else
        if (enforce_stell_sym) bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1)
        if (.not.enforce_stell_sym) bnserr = sum(abs(iBns(2:mn) - ofmn(2:mn))) / (mn - 1) + sum(abs(iBnc(1:mn) -&
            & efmn(1:mn))) / (mn)
      end if

      if (bnserr > gBntol) then

        LContinueFreeboundaryIterations = .true.

        select case (mfreeits)

        case (-2) ! mfreeits = -2  shall set plasma normal field at computational boundary

          inquire(file=trim(ext) // ".Vn", exist=exist)
          call fatal(.not.exist, __FILE__, "ext.Vn does not exist : cannot set vacuum field")

          if (myid == 0) then ! only myid = 0 reads in the vacuum field

            iBns(2:mn) = iVns(2:mn) - iBns(2:mn)! temporary storage of the total field
            if (.not.enforce_stell_sym) iBnc(1:mn) = iVnc(1:mn) - iBnc(1:mn)

            open(lunit, file=trim(ext) // ".Vn", status="old")
            read(lunit, *)lmn, lNfp
            read(lunit, *)(lim, lin, lRwc, lRws, lZwc, lZws, imn = 1, lmn)! control surface should confirm agreement
            read(lunit, *)lmn, lNfp, lItor, lGpol
            do imn = 1, lmn
              read(lunit, *)lim, lin, lgBc, lgBs
              do ii = 1, mn
                if (lim == im(ii) .and. lin * lNfp == in(ii)) then
                  iVns(ii) = lgBs
                  if (.not.enforce_stell_sym) iVnc(ii) = lgBc
                end if
              end do
            end do
            close(lunit)

          end if

          call broadcast(iVns(1:mn), mn, 0)! only required for ii > 1
          if (.not.enforce_stell_sym) then
            call broadcast(iVnc(1:mn), mn, 0)
          end if

          iBns(2:mn) = -iBns(2:mn) - iVns(2:mn)! updated vacuum field
          if (.not.enforce_stell_sym) iBnc(1:mn) = -iBnc(1:mn) - iVnc(1:mn)

          if (myid == 0) then
            write(ounit, '("xspech : " 10x " : oBns=[",999(es11.03,","))')iBns(1:mn)
            write(ounit, '("xspech : " 10x " : nBns=[",999(es11.03,","))')ofmn(1:mn)
            if (.not.enforce_stell_sym) then
              write(ounit, '("xspech : " 10x " : oBnc=[",999(es11.03,","))')iBnc(1:mn)
              write(ounit, '("xspech : " 10x " : nBnc=[",999(es11.03,","))')efmn(1:mn)
            end if
          end if

        case (-1) ! mfreeits = -1 shall set vacuum normal field at computational boundary

          iVns(2:mn) = iVns(2:mn) - iBns(2:mn)! total   normal field
          if (.not.enforce_stell_sym) iVnc(1:mn) = iVnc(1:mn) - iBnc(1:mn)
          iBns(2:mn) = ofmn(2:mn)! updated normal field due to plasma
          if (.not.enforce_stell_sym) iBnc(1:mn) = efmn(1:mn)
          iVns(2:mn) = iVns(2:mn) + iBns(2:mn)! updated vacuum field
          if (.not.enforce_stell_sym) iVnc(1:mn) = iVnc(1:mn) + iBnc(1:mn)

        case (0) ! mfreeits =  0

          call fatal(.true., __FILE__, "illegal mfreeits logic ")

        case (1:) ! mfreeits >  0

          select case (Lzerovac)

          case (0) ! Lzerovac = 0

            iBns(2:mn) = gBnbld * iBns(2:mn) + (1.0_rkind - gBnbld) * ofmn(2:mn)
            if (.not.enforce_stell_sym) iBnc(1:mn) = gBnbld * iBnc(1:mn) + (1.0_rkind - gBnbld) * efmn(1:mn)

          case (1) ! Lzerovac = 1

            iBns(2:mn) = ofmn(2:mn)! no blend
            if (.not.enforce_stell_sym) iBnc(1:mn) = efmn(1:mn)

            iVns(2:mn) = +iBns(2:mn)! update vacuum field to cancel plasma field on computational boundary
            if (.not.enforce_stell_sym) iVnc(1:mn) = +iBnc(1:mn)

          case default ! Lzerovac

            call fatal(.true., __FILE__, "invalid Lzerovac ")

          end select ! end select case( Lzerovac )

        end select ! end select case( mfreeits )

      else ! else of if( bnserr.gt.gBntol )

        LContinueFreeboundaryIterations = .false.

      end if

      cput = MPI_Wtime()

      if (myid == 0) then
        write(ounit, 1003)
        write(ounit, 1004)cput - cpus, nfreeboundaryiterations, mfreeits, gBntol, bnserr, cput - lastcpu
        write(ounit, 1003)
      end if

      1003 format("xspech : ", 10x, " : ")
      1004 format("xspech : ", f10.2, " : nfreeboundaryiterations = ", i6, " / ", i6.5, " ; gBntol = ", es8.1, " ; bnserr = ", &
          es12.5, " ; bnorml time = ", f10.2, "s ")

    end if

    call write_vector_potentials('W')! this writes vector potential to file

    ! if (myid == 0) then ! write restart file note that this is inside free-boundary iteration loop
    !   call wrtend ! write restart file save initial input
    ! end if

    if (LContinueFreeboundaryIterations .and. nfreeboundaryiterations < mfreeits) goto 9000
    if (Lfreebound == 1 .and. First_free_bound) goto 9000 ! going back to normal free_boundary calculation
  end subroutine spectre

  subroutine final_diagnostics
    real(rkind) :: cput

    integer :: iocons, llmodnp, vvol, iflag, cpu_id
    real :: sumI
    real(rkind), allocatable :: Bt00(:, :, :)

    call set_fftw_plan(Nt, Nz)
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

        call compute_rotational_transform(vvol, mn, lmns, Nt, Nz, iflag, diotadxup(0:1, -1:2, vvol))! stores lambda in a global variable.
      end do

      ! Broadcast
      do vvol = 1, Mvol
        cpu_id = WhichCpuID(vvol, ncpu)
        call broadcast(diotadxup(0:1, -1:2, vvol), 8, cpu_id)
        call broadcast(dlambdaout(1:lmns, vvol, 0:1), 2 * lmns, cpu_id)
      end do

    end if

    ! Computes the surface current at each interface for output

    call sallocate(Bt00, 1, Mvol, 0, 1, -1, 2, 0.0_rkind)

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      do iocons = 0, 1
        if (Lcoordinatesingularity .and. iocons == 0) cycle
        if (vvol == Nvol + 1 .and. iocons == 1) cycle
        ! Compute covariant magnetic field at interface
        call lbpol(vvol, Mvol, Nt, Nz, Bt00(1:Mvol, 0:1, -1:2), 0, iocons)

        ! Save covariant magnetic field at interface for output - computed in lbpol
        Btemn(1:mn, iocons, vvol) = efmn(1:mn)
        Btomn(1:mn, iocons, vvol) = ofmn(1:mn)
        Bzemn(1:mn, iocons, vvol) = cfmn(1:mn)
        Bzomn(1:mn, iocons, vvol) = sfmn(1:mn)
      end do
    end do

    ! Evaluate surface current
    do vvol = 1, Mvol - 1
      IPDt(vvol) = pi2 * (Bt00(vvol + 1, 0, 0) - Bt00(vvol, 1, 0))
    end do

    deallocate(Bt00)

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
        write(ounit, '("xspech : ",f10.2," : myid=",i3," ; Poincare plot ; odetol="es8.1" ; nPpts="i7" ":" nPtrj="24(i5"&
            &          ,")" ...")')cput - cpus, myid, odetol, nPpts, nPtrj(1:min(Mvol, 24))
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
      call broadcast(Btemn(1:mn, 0:1, vvol), mn * 2, llmodnp)! this is computed in lbpol
      call broadcast(Bzemn(1:mn, 0:1, vvol), mn * 2, llmodnp)
      call broadcast(Btomn(1:mn, 0:1, vvol), mn * 2, llmodnp)
      call broadcast(Bzomn(1:mn, 0:1, vvol), mn * 2, llmodnp)

      call broadcast(beltramierror(vvol, 1:9), 9, llmodnp)! this is computed in compute_beltrami_error

    end do

  end subroutine final_diagnostics

  subroutine ending
    real(rkind) :: cput, cpui, cpuo

    real(rkind) :: Ttotal, dcpu, ecpu
    character :: date * 8, time * 10

    cpui = MPI_Wtime()
    cpuo = cpui ! see macro expansion for begin

    cput = MPI_Wtime()
    dcpu = cput - cpus

    if (.false. .and. myid == 0) then

      Ttotal = 0.0_rkind

      write(ounit, '("ending : ",f10.2," : time spent in wrtend =",f10.2," ;")')dcpu, Twrtend
      Ttotal = Ttotal + Twrtend
      write(ounit, '("ending : ",f10.2," : time spent in readin =",f10.2," ;")')dcpu, Treadin
      Ttotal = Ttotal + Treadin

      ecpu = Ttotal - dcpu ! error in actual cpu time and calculated cpu time

      write(ounit, '("ending : ",f10.2," : Ttotal =",f10.2," s = "f8.2" m = "f6.2" h Timing Error = ",f10.2,"s = ",f10.2&
          &        ,"%")')dcpu, Ttotal / [1, 60, 3600], ecpu, 100 * ecpu / dcpu

    end if

    if (myid == 0) then
      call date_and_time(date, time)
      write(ounit, '("ending : ", 10x ," : ")')
      write(ounit, 1000)dcpu, myid, dcpu / [1, 60, 60 * 60, 24 * 60 * 60], date(1:4), date(5:6), date(7:8), time(1:2)&
          &, time(3:4), time(5:6), ext
      write(ounit, '("ending : ", 10x ," : ")')
    end if

    1000 format("ending : ", f10.2, " : myid=", i3, " ; completion ; time=", f10.2, "s = ", f8.2, "m = ", f6.2, "h = ", f5.2, &
        "d date= ", a4, "/", a2, "/", a2, " ; time= ", a2, ":", a2, ":", a2, "  ext = ", a60)
  end subroutine ending
end module simulation_mod
