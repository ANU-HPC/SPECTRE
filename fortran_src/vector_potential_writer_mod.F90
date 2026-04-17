module vector_potential_writer_mod
  use precisions_mod, only: rkind
  use numerical_mod, only:
  use io_mod, only: ounit, aunit
  use input_list_mod, only: Wmacros, Wwrite_vector_potentials, Nfp, Mpol, Ntor, Lrad
  use allglobal_mod, only: myid, cpus, ext, get_hidden, Mvol, mn, im, in, Ate, Aze, Ato, Azo
  use sphdf5_mod, only: write_vector_potential
  use memory_allocation_mod, only: sallocate, nallocate
  use errors_mod, only: fatal
  use mpi_collectives_mod, only: broadcast

  implicit none(type, external)

  private
  public :: write_vector_potentials

contains

  subroutine write_vector_potentials(writeorread)
    character, intent(in) :: writeorread

    real(rkind) :: cput

    logical :: exist

    integer :: ios
    integer :: vvol, oldMvol, oldMpol, oldNtor, oldmn, oldNfp, oldLrad, ii, jj, minLrad, llmodnp, ideriv, sumLrad
    integer, allocatable :: oldim(:), oldin(:)
    real(rkind), allocatable :: oldAte(:), oldAze(:), oldAto(:), oldAzo(:)
    real(rkind), allocatable :: allAte(:, :), allAze(:, :), allAto(:, :), allAzo(:, :)

    ideriv = 0 ! write vector potential and not derivative (?)

    select case (writeorread)
    case ('W') ! write vector potential harmonics to file
      ! determine total radial dimension: sum(Lrad(1:Mvol)+1)
      sumLrad = sum(Lrad(1:Mvol) + 1)

      call nallocate(allAte, 1, sumLrad, 1, mn)
      call nallocate(allAze, 1, sumLrad, 1, mn)
      call nallocate(allAto, 1, sumLrad, 1, mn)
      call nallocate(allAzo, 1, sumLrad, 1, mn)

      ! collect data in 2D arrays which can be written more conveniently
      sumLrad = 1
      do vvol = 1, Mvol

        ! TODO: MPI_allgather ????
        ! or at least selective via IsMyVolume()

        do ii = 1, mn ! loop over Fourier harmonics
          allAte(sumLrad:sumLrad + Lrad(vvol), ii) = Ate(vvol, ideriv, ii)%s(0:Lrad(vvol))
          allAze(sumLrad:sumLrad + Lrad(vvol), ii) = Aze(vvol, ideriv, ii)%s(0:Lrad(vvol))
          allAto(sumLrad:sumLrad + Lrad(vvol), ii) = Ato(vvol, ideriv, ii)%s(0:Lrad(vvol))
          allAzo(sumLrad:sumLrad + Lrad(vvol), ii) = Azo(vvol, ideriv, ii)%s(0:Lrad(vvol))
        end do
        sumLrad = sumLrad + Lrad(vvol) + 1
      end do

      ! determine total radial dimension: sum(Lrad(1:Mvol)+1)
      sumLrad = sum(Lrad(1:Mvol) + 1)

      ! HDF5 calls have to be done from all CPUs when using the collective API
      call write_vector_potential(sumLrad, allAte, allAze, allAto, allAzo)

      ! clean up after yourself
      deallocate(allAte)
      deallocate(allAze)
      deallocate(allAto)
      deallocate(allAzo)

    case ('R') ! read potential from file: interpolate onto new radial grid
      if (myid == 0) then

        inquire(file=trim(get_hidden(ext)) // ".sp.A", exist=exist)

        if (.not.exist) then
          write(ounit, '("write_vector_potentials : ",f10.2," : myid=",i3," ; error ; .ext.sp.A does not exist ;")') &
              cput - cpus, myid
          goto 9998
        end if

        open(aunit, file=trim(get_hidden(ext)) // ".sp.A", status="old", form="unformatted", iostat=ios)! this will
        ! contain initial guess for vector potential

        if (ios /= 0) then
          write(ounit, '("write_vector_potentials : ",f10.2," : myid=",i3," ; error ; opening .ext.sp.A ;")')cput - cpus, myid
          goto 9997
        end if

        read(aunit, iostat=ios)oldMvol, oldMpol, oldNtor, oldmn, oldNfp ! these are the "old" resolution parameters

        if (ios /= 0) then
          write(ounit, '("write_vector_potentials : ",f10.2," : myid=",i3, &
              & "  error  reading oldMvol, oldMpol, oldNtor, oldmn, oldNfp")')cput - cpus, myid
          goto 9997
        end if

        if (oldNfp /= Nfp) then
          write(ounit, '("write_vector_potentials : ",f10.2," : myid=",i3," ; error ; inconsistent Nfp ; ")')cput - cpus, myid
          goto 9997
        end if
        if (oldMvol /= Mvol) then
          write(ounit, '("write_vector_potentials : ",f10.2," : myid=",i3," ; error ; inconsistent Mvol ;")')cput - cpus, myid
          goto 9997
        end if

        call sallocate(oldim, 1, oldmn, 0)
        call sallocate(oldin, 1, oldmn, 0)

        read(aunit, iostat=ios)oldim(1:oldmn)
        read(aunit, iostat=ios)oldin(1:oldmn)

        do vvol = 1, oldMvol

          read(aunit, iostat=ios)oldLrad

          minLrad = min(oldLrad, Lrad(vvol))

          call sallocate(oldAte, 0, oldLrad, 0.0_rkind)
          call sallocate(oldAze, 0, oldLrad, 0.0_rkind)
          call sallocate(oldAto, 0, oldLrad, 0.0_rkind)
          call sallocate(oldAzo, 0, oldLrad, 0.0_rkind)

          do jj = 1, oldmn

            read(aunit, iostat=ios)oldAte(0:oldLrad)
            read(aunit, iostat=ios)oldAze(0:oldLrad)
            read(aunit, iostat=ios)oldAto(0:oldLrad)
            read(aunit, iostat=ios)oldAzo(0:oldLrad)

            do ii = 1, mn ! compare Fourier harmonic with old
              if (im(ii) == oldim(jj) .and. in(ii) == oldin(jj)) then
                Ate(vvol, ideriv, ii)%s(0:minLrad) = oldAte(0:minLrad)
                Aze(vvol, ideriv, ii)%s(0:minLrad) = oldAze(0:minLrad)
                Ato(vvol, ideriv, ii)%s(0:minLrad) = oldAto(0:minLrad)
                Azo(vvol, ideriv, ii)%s(0:minLrad) = oldAzo(0:minLrad)
              end if
            end do

          end do

          deallocate(oldAte)
          deallocate(oldAze)
          deallocate(oldAto)
          deallocate(oldAzo)

        end do

        deallocate(oldim)
        deallocate(oldin)

        9997 continue

        close(aunit)

        9998 continue

      end if

      do vvol = 1, Mvol

        llmodnp = 0 ! this node contains the information that is to be broadcast

        do ii = 1, mn
          call broadcast(Ate(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, llmodnp)
          call broadcast(Aze(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, llmodnp)
        end do
        !if( .not.enforce_stell_sym ) then
        do ii = 1, mn
          call broadcast(Ato(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, llmodnp)
          call broadcast(Azo(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, llmodnp)
        end do
        !endif

      end do

    case default
      call fatal(.true., __FILE__, "invalid writeorread flag supplied on input")
    end select
  end subroutine write_vector_potentials
end module vector_potential_writer_mod
