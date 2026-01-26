module freeb_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind
  use io_mod, only: stdout, lunit
  use input_list_mod, only: enforce_stell_sym, &
      Lconstraint, &
      mu, &
      nvol, &
      Lrad, &
      Lcheck, &
      vcNt, &
      vcNz, &
      Lvcgrid, &
      Igeometry, &
      vcasingtol, &
      vcasingits, &
      vcasingper, &
      vcasingeps, &
      nfp
  use allglobal_mod, only: myid, &
      ncpu, &
      cpus, &
      mn, &
      mvol, &
      ijreal, &
      ijimag, &
      first_free_bound, &
      Pbxyz, &
      Jxyz, &
      prevcstride, &
      Nt, &
      Nz, &
      virtualcasingfactor, &
      im, &
      in, &
      Ate, &
      Aze, &
      Ato, &
      Azo, &
      iRbc, &
      iZbc, &
      iRbs, &
      iZbs, &
      TT, &
      globaljk, &
      Nxyz, &
      Dxyz, &
      MPI_COMM_SPEC, &
      efmn, &
      ofmn, &
      cfmn, &
      sfmn
  use errors_mod, only: fatal
  use constants_mod, only: pi, pi2
  use memory_allocation_mod, only: sallocate
  use mpi_collectives_mod, only: broadcast
  use numerical_mod, only: tfft

  implicit none(type, external)

  private
  public :: compute_bnorm_plasmab

  external :: dcuhre

contains

  subroutine compute_bnorm_plasmab(mn, Ntz, efmn, ofmn)
    integer, intent(in) :: mn, Ntz
    real(rkind), intent(out) :: efmn(1:mn), ofmn(1:mn)

    integer :: lvol, Lparallel, ii, jj, kk, jk, ll, kkmodnp, jkmodnp, ifail, id01daf, nvccalls, icasing
    real(rkind) :: zeta, teta, gBn

    real(rkind) :: Bxyz(1:Ntz, 1:3), distance(1:Ntz)

    real(rkind) :: absvcerr, relvcerr, resulth, resulth2, resulth4, deltah4h2, deltah2h
    integer :: vcstride, Nzwithsym, vcNznfp, ierr

    Lparallel = 1 ! controls choice of parallelization see below

    ijreal(1:Ntz) = 0.0_rkind ! normal plasma field
    ijimag(1:Ntz) = 0.0_rkind

    vcNznfp = vcNz * nfp

    ! In stellerator symmetric geometry, the virtual casing field also exhibits symmetriy which we can exploit
    if (enforce_stell_sym) then
      ! we have to compute approximately half points due to symmetry, but including the middle and the edge points! (Closed interval
      ! [0, pi])
      Nzwithsym = min(Nz, (Nz + 3) / 2)
    else
      Nzwithsym = Nz
    end if

    if ( Lvcgrid == 1 ) then
      ! Precompute Jxyz(:,1:3) and the corresponding positions on the high resolution plasma boundary
      Pbxyz(:, 1:3) = 0.0_rkind
      Jxyz(:, 1:3) = 0.0_rkind

      !$OMP PARALLEL DO SHARED(Pbxyz, Jxyz) PRIVATE(jk, jj, kk, teta, zeta) COLLAPSE(2)
      do kk = 0, vcNznfp - 1
        do jj = 0, vcNt - 1
          zeta = kk * pi2 / vcNznfp
          teta = jj * pi2 / vcNt
          jk = 1 + jj + kk * vcNt

          ! Each MPI rank only computes every a 1/ncpu surfacecurrent() calls
          select case ( Lparallel )
          case ( 0 ) ! Lparallel = 0
            if ( myid /= modulo(kk, ncpu)) cycle
          case ( 1 ) ! Lparallel = 1
            if ( myid /= modulo(jk - 1, ncpu)) cycle
          case default ! Lparallel
            call fatal(.true., __FILE__, "invalid Lparallel in parallelization loop")
          end select

          call surfacecurrent(teta, zeta, Pbxyz(jk, 1:3), Jxyz(jk, 1:3))
        end do
      end do

      !! NOTE: when ran through python wrappers, mpi_allreduce with inplace does not work !!!
      ! MPI reductions for positions and currents to accumulate them on all ranks (valid because initialized to zero)
      ! and Broadcast the total currents and evaluation points back to all ranks
      call mpi_allreduce(MPI_IN_PLACE, Pbxyz(:, 1:3), 3 * vcNt * vcNznfp, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_SPEC, ierr)
      if (ierr /= MPI_SUCCESS) then
        call fatal(.true., __FILE__, "error in mpi_allreduce for Pbxyz")
      end if

      call mpi_allreduce(MPI_IN_PLACE, Jxyz(:, 1:3), 3 * vcNt * vcNznfp, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_SPEC, ierr)
      if (ierr /= MPI_SUCCESS) then
        call fatal(.true., __FILE__, "error in mpi_allreduce for Jxyz")
      end if

      ! iterate over resolutions of the virtual casing grid to get an estimate of the accuracy. Write the result into ijimag
      deltah2h = 1e12
      !> The surface currents get precomputed at all points, but maybe we don't have to integrate over the whole grid.
      !> Start with a large stride over the plasmaboundary (= how many points to skip at each step), then get progressively finer.
      !> Do at least vcasingits iterations (sqrt(vcasingits) resolution per dimension) and halve the stride until the accuracy is
      !reached.
      prevcstride = min(prevcstride, INT(0.5 * log(real(vcNt * vcNznfp / vcasingits)) / log(2.0)))
      prevcstride = max(prevcstride, 1)!> At least one step in the resolution cascade is required to determine an estimate of the current accuracy.
      do vcstride = prevcstride, 0, -1
        !$OMP PARALLEL DO SHARED(Dxyz, Nxyz, Pbxyz, Jxyz, ijreal, ijimag) FIRSTPRIVATE(jk, gBn) COLLAPSE(2)
        do kk = 0, Nzwithsym - 1
          do jj = 0, Nt - 1
            jk = 1 + jj + kk * Nt

            ! kk = 0 terms are also symmetryic (relevant e.g. for tokamaks)
            if ( (Nz == 1) .and. (jj > ( (Nt + 1) / 2) ) ) cycle

            ! Each MPI rank only computes every a 1/ncpu surfacecurrent() calls
            ! Identical MPI parallelization scheme as for Lvcgrid=0
            select case ( Lparallel )
            case ( 0 ) ! Lparallel = 0
              if ( myid /= modulo(kk, ncpu)) cycle
            case ( 1 ) ! Lparallel = 1
              if ( myid /= modulo(jk - 1, ncpu)) cycle
            case default ! Lparallel
              call fatal(.true., __FILE__, "invalid Lparallel in parallelization loop")
            end select ! end of select case( Lparallel )

            gBn = 0.0_rkind
            globaljk = jk ! only to check against the precomputed values against dvcfield

            call casinggrid(Dxyz(1:3, jk), Nxyz(1:3, jk), Pbxyz, Jxyz, 2 ** vcstride, gBn)

            ijimag(jk) = ijreal(jk)! previous solution (lower resolution)
            ijreal(jk) = gBn ! current solution (higher resolution)
          end do
        end do

        deltah4h2 = deltah2h
        deltah2h = maxval(abs(ijimag - ijreal))! mean delta between the h and h/2 solutions

        ! Order of the integration method: log(deltah4h2/deltah2h)/log(2.0) = 1
        absvcerr = deltah2h
        relvcerr = 2 * deltah2h / maxval(abs(ijreal))! overestimate the relative error by a factor of two
        if (myid == 0) then
          write(stdout, '("bnorml : ", 10x ," : relvcerr = ",es13.5," ; absvcerr = ",es13.5," ; resolution reduction = ",i8)')&
              relvcerr, absvcerr, 2 ** vcstride
          ! print *, "Convergence order: ",  log(deltah4h2/deltah2h)/log(2.0)
        end if
        ! Tolerance was already reached, exit the loop. (Different threads may exit at different times)
        if (absvcerr < vcasingtol .and. relvcerr < vcasingtol) then
          prevcstride = vcstride
          exit
        end if
      end do

    else
      do kk = 0, Nzwithsym - 1
        zeta = kk * pi2 / Nz

        do jj = 0, Nt - 1
          teta = jj * pi2 / Nt
          jk = 1 + jj + kk * Nt

          ! kk = 0 terms are also symmetryic (relevant e.g. for tokamaks)
          if ( (Nz == 1) .and. (jj > ( (Nt + 1) / 2) ) ) cycle

          globaljk = jk ! this is global; passed through to vcintegrand & casing

          select case ( Lparallel ) ! perform in parallel
          case ( 0 )
            if ( myid /= modulo(kk, ncpu)) cycle
          case ( 1 )
            if ( myid /= modulo(jk - 1, ncpu)) cycle ! 11 Oct 12, this is a weird parallelization, but perhaps better exploits all available cpus;
          case default
            call fatal(.true., __FILE__, "invalid Lparallel in parallelization loop")
          end select

          call casing(teta, zeta, gBn, icasing)
          ijreal(jk) = gBn

        end do
      end do
    end if

    1001 format("bnorml : ", 10x, " : "a1" : (t,z) = ("f8.4","f8.4" ) ; gBn=", f23.15, " ; ":" error =", f23.15" ;")

    if ( myid == 0 .and. Lcheck == 6 ) then ! THIS WAS CORRUPTED; see before 14 Apr 17 for complete source;
      close(lunit)
    end if

    do kk = 0, Nz - 1

      kkmodnp = modulo(kk, ncpu)

      select case ( Lparallel )

      case ( 0 )
        call broadcast(ijreal(1 + kk * Nt:Nt + kk * Nt), Nt, kkmodnp)
      case ( 1 )

        do jj = 0, Nt - 1

          jk = 1 + jj + kk * Nt

          jkmodnp = modulo(jk - 1, ncpu)

          call broadcast(ijreal(jk), jkmodnp)
        end do

      case default

        call fatal(.true., __FILE__, "invalid Lparallel for broadcasting")

      end select

    end do

    ! In stellerator symmetric geometry, the virtual casing field also exhibits symmetriy which we can exploit
    if (enforce_stell_sym) then
      ! We are dealing with half open intervals in theta and phi [0, 2pi[ so we need to skip the first row and column when mirroring
      if (Nz == 1) then
        do jj = 0, Nt / 2
          jk = 1 + jj
          ijreal(Ntz + 1 - jk) = -ijreal(jk + 1)
        end do
      end if

      do kk = 0, Nzwithsym - 2
        do jj = 0, Nt - 1
          jk = 1 + jj + kk * Nt

          ijreal(Ntz + 1 - jk) = -ijreal(jk + Nt + 1)
        end do
      end do
    end if

    ijreal(1:Ntz) = ijreal(1:Ntz) * virtualcasingfactor
    ijimag(1:Ntz) = 0.0_rkind

    call tfft(ijreal(1:Ntz), ijimag(1:Ntz), mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), ifail)! Fourier decompose normal field

  end subroutine compute_bnorm_plasmab

  !> Compute the external magnetic field using the virtual casing method
  subroutine casing(teta, zeta, gBn, icasing)
    real(rkind), intent(in) :: teta, zeta ! theta and zeta evaluation point on the computational boundary, corresponding to the cartesian evaluation point Dxyz(1:3,globaljk)
    real(rkind), intent(out) :: gBn ! B.n magnetic field
    integer, intent(out) :: icasing

    integer, parameter :: Ndim = 2, Nfun = 1

    integer :: ldim, lfun, minpts, maxpts, Lrwk, idcuhre, irestart, funcls, key, num, maxsub
    real(rkind) :: integrals(1:Nfun), low(1:Ndim), upp(1:Ndim), labs, lrel, absest(1:Nfun)
    real(rkind), allocatable :: rwk(:)

    gBn = 0.0_rkind

    ldim = Ndim ! number of dimensions of the integral; surface integral => number of dimensions = 2;

    low(1) = teta - vcasingper * pi
    upp(1) = teta + vcasingper * pi
    low(2) = zeta - vcasingper * pi
    upp(2) = zeta + vcasingper * pi

    key = 0

    minpts = vcasingits ! minimum number of function evaluations, provided on input

    num = 65 ! see documentation for dcuhre

    !  maxpts = max( minpts, 3 * num )

    maxpts = 16777216

    lfun = Nfun ! number of functions to be integrated, require three components of magnetic field, Bx, By and Bz, and their derivatives wrt x,y,z;

    labs = vcasingtol
    lrel = vcasingtol ! absolute and relative accuracy requested, vcasingtol is an input parameter

    maxsub = ( maxpts - num ) / ( 2 * num ) + 1

    Lrwk = maxsub * ( 2 * Ndim + 2 * Nfun + 2 ) + 17 * Nfun + 1

    call sallocate(rwk, 1, Lrwk, 0.0_rkind)

    irestart = 0
    funcls = 0

    do ! will continually call until satisfactory accuracy has been achieved

      idcuhre = 1

      call dcuhre(ldim, lfun, low(1:Ndim), upp(1:Ndim), minpts, maxpts, dvcfield, labs, lrel, key, Lrwk, irestart, &
          integrals(1:lfun), absest(1:lfun), funcls, idcuhre, rwk(1:Lrwk))

      gBn = integrals(1)

      select case ( idcuhre )
      case (0)
        exit
      case (1)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "maxpts too smal;        "
        !exit
      case (2)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "illegal key;            "
        exit
      case (3)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "illegal Ndim;           "
        exit
      case (4)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "key.eq.1 & Ndim.ne.2;   "
        exit
      case (5)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "key.eq.2 & Ndim.ne.3;   "
        exit
      case (6)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "numfun < 1;             "
        exit
      case (7)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "volume is zero;         "
        exit
      case (8)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "maxpts < 3*NUM;         "
        exit
      case (9)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "maxpts < minpts;        "
        exit
      case (10)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "epsabs < 0 & epsrel < 0;"
        exit
      case (11)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "NW is too small;        "
        exit
      case (12)
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "illegal irestart;       "
        exit
      case default
        write(stdout, 1001)myid, Dxyz(1:3, globaljk), gBn, absest(1:Nfun), idcuhre, minpts, maxpts, "tryin' to kill me?      "
        exit
      end select

      maxpts = 2 * maxpts
      minpts = funcls
      irestart = 0

      maxsub = ( maxpts - num ) / ( 2 * num ) + 1

      Lrwk = maxsub * ( 2 * Ndim + 2 * Nfun + 2 ) + 17 * Nfun + 1

      deallocate(rwk)

      call sallocate(rwk, 1, Lrwk, 0.0_rkind)
    end do

    1001 format("casing : myid=", i3, " ; [x,y,z]=["es10.2" ,"es10.2" ,"es10.2" ]; gBn="es12.4" , ",&
        "err="es8.0" ; ifail="i3" ; min/max calls="2i12" ; "a24)

    icasing = idcuhre ! this is an error flag returned by casing

    deallocate(rwk)

  end subroutine casing

  subroutine casinggrid(xyz, ntz, Pbxyz, Jxyz, vcstride, gBn)

    real(rkind), intent(in) :: xyz(1:3)! arbitrary location; Cartesian
    real(rkind), intent(in) :: ntz(1:3)! surface normal on the computational boundary; Cartesian
    real(rkind), intent(in) :: Pbxyz(1:vcNz * Nfp * vcNt, 1:3), Jxyz(1:vcNz * Nfp * vcNt, 1:3)
    integer, intent(in) :: vcstride
    real(rkind), intent(out) :: gBn ! B.n on the computational boundary

    real(rkind) :: rr(1:3), distance(1:3), jj(1:3), Bxyz(1:3), accumulator, firstorderfactor
    real(rkind) :: gBnlocal ! Local accumulator so we don't directly sum into the destination pointer (Avoid cache thrashing due to false share)
    real(rkind) :: c, t, y ! Helper variables for the Kahan summation
    integer :: plasmaNtz, jk ! plasmaNtz = SIZE(Pbxyz, 1)
    plasmaNtz = vcNz * Nfp * vcNt
    gBnlocal = 0.0_rkind
    c = 0.0_rkind

    ! loop over the high resolution plasma boundary (inner boundary for virtual casing)
    do jk = 1, plasmaNtz, vcstride

      ! distance vector between point on computational boundary and point on plasma boundary
      rr = xyz - Pbxyz(jk, 1:3)
      jj = Jxyz(jk, 1:3)
      distance(2) = sum(rr * rr) + vcasingeps ** 2
      distance(1) = sqrt(distance(2))
      distance(3) = distance(1) * distance(2)

      firstorderfactor = ( 1.0_rkind + 3.0_rkind * vcasingeps ** 2 / distance(2)) / distance(3)
      Bxyz(1:3) = (/ jj(2) * rr(3) - jj(3) * rr(2), &
          jj(3) * rr(1) - jj(1) * rr(3), &
          jj(1) * rr(2) - jj(2) * rr(1)/)

      ! Accumulate B.r/r^3 contributions
      ! Kahan Summation algorithm
      y = sum(Bxyz * ntz) * firstorderfactor - c ! Correct the next value with the compensation
      t = gBnlocal + y ! Add the corrected value to the sum
      c = (t - gBnlocal) - y ! Update compensation
      gBnlocal = t ! Update the running total
    end do
    gBn = gBnlocal * pi2 * pi2 / (plasmaNtz / vcstride)
    return

  end subroutine casinggrid

  subroutine surfacecurrent(teta, zeta, pxyz, jj)

    real(rkind), intent(in) :: teta, zeta ! theta and zeta coordinates on the plasma boundary
    real(rkind), intent(out) :: pxyz(1:3)! Position on the plasma boundary
    real(rkind), intent(out) :: jj(1:3)! Cartesian surface current components on the plasma boundary

    integer :: ii, mi, ni, ll, ideriv
    real(rkind) :: dR(0:3), dZ(0:3), gBut, gBuz, gtt, gtz, gzz, sqrtg, Blt, Blz, czeta, szeta, arg, carg, sarg

    real(rkind) :: XXt, XXz, YYt, YYz, ZZt, ZZz, ds
    dR(0:3) = 0.0_rkind
    ideriv = 0
    gBut = 0.0_rkind
    gBuz = 0.0_rkind
    dZ(0:3) = 0.0_rkind

    select case ( Igeometry )

    case ( 1 )

      if ( enforce_stell_sym ) then

        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)! loop over Fourier modes; construct surface current; slow transform required as position is arbitrary;

          arg = mi * teta - ni * zeta
          carg = cos(arg)
          sarg = sin(arg)

          dR(0) = dR(0) + ( iRbc(ii, Nvol)) * carg
          dR(1) = dR(1) + ( ( iRbc(ii, Mvol) - iRbc(ii, Nvol)) * carg ) * 0.5_rkind
          dR(2) = dR(2) + mi * (- ( iRbc(ii, Nvol)) * sarg )
          dR(3) = dR(3) - ni * (- ( iRbc(ii, Nvol)) * sarg )

          ! dZ(0) = dZ(0) +                                                       (                 iZbs(ii,Nvol) ) * sarg
          ! dZ(1) = dZ(1) +      (                                                ( iZbs(ii,Mvol) - iZbs(ii,Nvol) ) * sarg ) *
          ! 0.5_rkind
          ! dZ(2) = dZ(2) + mi * (                                                (                 iZbs(ii,Nvol) ) * carg )
          ! dZ(3) = dZ(3) - ni * (                                                (                 iZbs(ii,Nvol) ) * carg )

          do ll = 0, Lrad(Mvol)
            gBut = gBut - ( Aze(Mvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 0, 1)! contravariant; Jacobian comes later;
            gBuz = gBuz + ( Ate(Mvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 0, 1)
          end do

        end do

      else

        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)! loop over Fourier modes; construct surface current; slow transform required as position is arbitrary;

          arg = mi * teta - ni * zeta
          carg = cos(arg)
          sarg = sin(arg)

          dR(0) = dR(0) + iRbc(ii, Nvol) * carg + iRbs(ii, Nvol) * sarg
          dR(1) = dR(1) + ( ( iRbc(ii, Mvol) - iRbc(ii, Nvol)) * carg + ( iRbs(ii, Mvol) - iRbs(ii, Nvol)) * sarg ) * 0.5_rkind
          dR(2) = dR(2) + mi * (-iRbc(ii, Nvol) * sarg + iRbs(ii, Nvol) * carg )
          dR(3) = dR(3) - ni * (-iRbc(ii, Nvol) * sarg + iRbs(ii, Nvol) * carg )

          ! dZ(0) = dZ(0) +                              iZbc(ii,Nvol)   * carg +                   iZbs(ii,Nvol)   * sarg
          ! dZ(1) = dZ(1) +      (   (   iZbc(ii,Mvol) - iZbc(ii,Nvol) ) * carg + ( iZbs(ii,Mvol) - iZbs(ii,Nvol) ) * sarg ) *
          ! 0.5_rkind
          ! dZ(2) = dZ(2) + mi * ( -                     iZbc(ii,Nvol)   * sarg +                   iZbs(ii,Nvol)   * carg )
          ! dZ(3) = dZ(3) - ni * ( -                     iZbc(ii,Nvol)   * sarg +                   iZbs(ii,Nvol)   * carg )

          do ll = 0, Lrad(Mvol)
            gBut = gBut - ( Aze(Mvol, ideriv, ii)%s(ll) * carg + Azo(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 0, 1)! contravariant; Jacobian comes later;
            gBuz = gBuz + ( Ate(Mvol, ideriv, ii)%s(ll) * carg + Ato(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 0, 1)
          end do

        end do

      end if

    case ( 2 )

      call fatal(.true., __FILE__, "virtual casing under construction for cylindrical geometry")

    case ( 3 )

      if ( enforce_stell_sym ) then

        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)! loop over Fourier modes; construct surface current; slow transform required as position is arbitrary;

          arg = mi * teta - ni * zeta
          carg = cos(arg)
          sarg = sin(arg)

          dR(0) = dR(0) + ( iRbc(ii, Nvol)) * carg
          dR(1) = dR(1) + ( ( iRbc(ii, Mvol) - iRbc(ii, Nvol)) * carg ) * 0.5_rkind
          dR(2) = dR(2) + mi * (- ( iRbc(ii, Nvol)) * sarg )
          dR(3) = dR(3) - ni * (- ( iRbc(ii, Nvol)) * sarg )

          dZ(0) = dZ(0) + ( iZbs(ii, Nvol)) * sarg
          dZ(1) = dZ(1) + ( ( iZbs(ii, Mvol) - iZbs(ii, Nvol)) * sarg ) * 0.5_rkind
          dZ(2) = dZ(2) + mi * ( ( iZbs(ii, Nvol)) * carg )
          dZ(3) = dZ(3) - ni * ( ( iZbs(ii, Nvol)) * carg )

          if (first_free_bound) then
            do ll = 0, Lrad(Nvol)! 1 is for outside thr volume
              gBut = gBut - ( Aze(Nvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 1, 1)! contravariant; Jacobian comes later;
              gBuz = gBuz + ( Ate(Nvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 1, 1)
            end do
          else
            do ll = 0, Lrad(Mvol)
              gBut = gBut - ( Aze(Mvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 0, 1)! contravariant; Jacobian comes later;
              gBuz = gBuz + ( Ate(Mvol, ideriv, ii)%s(ll) * carg ) * TT(ll, 0, 1)
            end do
          end if

        end do
      else
        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)! loop over Fourier modes; construct surface current; slow transform required as position is arbitrary;

          arg = mi * teta - ni * zeta
          carg = cos(arg)
          sarg = sin(arg)

          dR(0) = dR(0) + iRbc(ii, Nvol) * carg + iRbs(ii, Nvol) * sarg
          dR(1) = dR(1) + ( ( iRbc(ii, Mvol) - iRbc(ii, Nvol)) * carg + ( iRbs(ii, Mvol) - iRbs(ii, Nvol)) * sarg ) * 0.5_rkind
          dR(2) = dR(2) + mi * (-iRbc(ii, Nvol) * sarg + iRbs(ii, Nvol) * carg )
          dR(3) = dR(3) - ni * (-iRbc(ii, Nvol) * sarg + iRbs(ii, Nvol) * carg )

          dZ(0) = dZ(0) + iZbc(ii, Nvol) * carg + iZbs(ii, Nvol) * sarg
          dZ(1) = dZ(1) + ( ( iZbc(ii, Mvol) - iZbc(ii, Nvol)) * carg + ( iZbs(ii, Mvol) - iZbs(ii, Nvol)) * sarg ) * 0.5_rkind
          dZ(2) = dZ(2) + mi * (-iZbc(ii, Nvol) * sarg + iZbs(ii, Nvol) * carg )
          dZ(3) = dZ(3) - ni * (-iZbc(ii, Nvol) * sarg + iZbs(ii, Nvol) * carg )

          if (first_free_bound) then
            do ll = 0, Lrad(Mvol)! omit the possible current sheet due to a jump in tangential field at the plasma boundary; Zhu 20190603;
              gBut = gBut - ( Aze(Mvol, ideriv, ii)%s(ll) * carg + Azo(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 1, 1)! contravariant; Jacobian comes later;
              gBuz = gBuz + ( Ate(Mvol, ideriv, ii)%s(ll) * carg + Ato(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 1, 1)
            end do
          else
            do ll = 0, Lrad(Mvol)
              gBut = gBut - ( Aze(Mvol, ideriv, ii)%s(ll) * carg + Azo(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 0, 1)! contravariant; Jacobian comes later;
              gBuz = gBuz + ( Ate(Mvol, ideriv, ii)%s(ll) * carg + Ato(Mvol, ideriv, ii)%s(ll) * sarg ) * TT(ll, 0, 1)
            end do
          end if

        end do

      end if

    end select

    select case ( Igeometry )

    case ( 1 )

      gtt = 1.0_rkind + dR(2) * dR(2)
      gtz = dR(2) * dR(3)
      gzz = 1.0_rkind + dR(3) * dR(3)

      sqrtg = dR(1)

    case ( 2 )

      call fatal(.true., __FILE__, "virtual casing under construction for cylindrical geometry")

    case ( 3 )

      gtt = dR(2) * dR(2) + dZ(2) * dZ(2)
      gtz = dR(2) * dR(3) + dZ(2) * dZ(3)
      gzz = dR(3) * dR(3) + dZ(3) * dZ(3) + dR(0) * dR(0)

      sqrtg = dR(0) * ( dZ(1) * dR(2) - dR(1) * dZ(2))

    end select

    Blt = ( gBut * gtt + gBuz * gtz ) / sqrtg
    Blz = ( gBut * gtz + gBuz * gzz ) / sqrtg

    select case ( Igeometry )

    case ( 1 )

      pxyz(1:3) = (/ teta, zeta, dR(0)/)
      XXt = 1.0_rkind
      XXz = 0.0_rkind
      YYt = 0.0_rkind
      YYz = 1.0_rkind
      ZZt = dR(2)
      ZZz = dR(3)

    case ( 2 )

      call fatal(.true., __FILE__, "virtual casing under construction for cylindrical geometry")

    case ( 3 )

      czeta = cos(zeta)
      szeta = sin(zeta)

      pxyz(1:3) = (/ dR(0) * czeta, dR(0) * szeta, dZ(0)/)
      XXt = dR(2) * czeta
      XXz = dR(3) * czeta - dR(0) * szeta
      YYt = dR(2) * szeta
      YYz = dR(3) * szeta + dR(0) * czeta
      ZZt = dZ(2)
      ZZz = dZ(3)

    end select

    jj(1:3) = (/ Blz * XXt - Blt * XXz, &
        Blz * YYt - Blt * YYz, &
        Blz * ZZt - Blt * ZZz /)

    return

  end subroutine surfacecurrent

  subroutine dvcfield(Ndim, tz, Nfun, vcintegrand)! differential virtual-casing field: format is fixed by NAG

    integer, intent(in) :: Ndim, Nfun
    real(rkind), intent(in) :: tz(1:Ndim)
    real(rkind), intent(out) :: vcintegrand(1:Nfun)! integrand; components of magnetic field due to plasma currents in Cartesian coordinates;

    real(rkind) :: jj(1:3), & ! Cartesian surface current components on the plasma boundary
        pxyz(1:3), & ! Position on the plasma boundary
        rr(1:3), & ! distance vector between evaluation point on the computational boundary and the current surface point
        distance(1:3), & ! powers of distance |(x-x')|, |(x-x')|^2, |(x-x')|^3 including a small eps to avoid division by zero
        Bxyz(1:3), firstorderfactor

    call surfacecurrent(tz(1), tz(2), pxyz, jj)

    ! distance vector between point on computational boundary and point on plasma boundary
    rr(1:3) = Dxyz(1:3, globaljk) - pxyz(1:3)

    distance(2) = sum(rr(1:3) * rr(1:3)) + vcasingeps ** 2
    distance(1) = sqrt(distance(2))
    distance(3) = distance(1) * distance(2)

    firstorderfactor = ( 1.0_rkind + 3.0_rkind * vcasingeps ** 2 / distance(2)) / distance(3)

    Bxyz(1:3) = (/ jj(2) * rr(3) - jj(3) * rr(2), &
        jj(3) * rr(1) - jj(1) * rr(3), &
        jj(1) * rr(2) - jj(2) * rr(1)/)

    vcintegrand(1) = sum(Bxyz(1:3) * Nxyz(1:3, globaljk)) * firstorderfactor

    ! vcintegrand( 4) = - 3.0_rkind * vcintegrand(1) * rr(1) / distance(2) ! dBxdx ! need to divide by distance(3)
    !   vcintegrand( 5) = - 3.0_rkind * vcintegrand(1) * rr(2) / distance(2) - jj(3) ! dBxdy
    !   vcintegrand( 6) = - 3.0_rkind * vcintegrand(1) * rr(3) / distance(2) + jj(2) ! dBxdz
    !
    !   vcintegrand( 7) = - 3.0_rkind * vcintegrand(2) * rr(1) / distance(2) + jj(3) ! dBydx
    !   vcintegrand( 8) = - 3.0_rkind * vcintegrand(2) * rr(2) / distance(2)         ! dBydy
    !   vcintegrand( 9) = - 3.0_rkind * vcintegrand(2) * rr(3) / distance(2) - jj(1) ! dBydz
    !
    !   vcintegrand(10) = - 3.0_rkind * vcintegrand(3) * rr(1) / distance(2) - jj(2) ! dBzdx
    !   vcintegrand(11) = - 3.0_rkind * vcintegrand(3) * rr(2) / distance(2) + jj(1) ! dBzdy
    !   vcintegrand(12) = - 3.0_rkind * vcintegrand(3) * rr(3) / distance(2)         ! dBzdz
    return

  end subroutine dvcfield
end module freeb_mod
