! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module packing_mod
  use precisions_mod, only: rkind
  use numerical_mod, only:
  use input_list_mod, only: Wpackxi, Igeometry, Ntor, Nvol, enforce_stell_sym, Wpackab, Lrad
  use allglobal_mod, only: psifactor, mn, Ate, Aze, Ato, Azo
  use coordinate_mod, only: rzaxis

  implicit none(type, external)

  private
  public :: packab, packxi

contains

  subroutine packab(packorunpack, lvol, NN, solution, ideriv)
    character, intent(in) :: packorunpack
    integer, intent(in) :: lvol, NN, ideriv
    real(rkind) :: solution(1:NN)

    integer :: ii, ll, id, llrad

    llrad = Lrad(lvol)! shorthand

    select case (packorunpack)

    case ('U')
      if (enforce_stell_sym) then

        do ii = 1, mn
          do ll = 0, llrad
            id = Ate(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Ate(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Ate(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
            id = Aze(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Aze(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Aze(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
            Ato(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            Azo(lvol, ideriv, ii)%s(ll) = 0.0_rkind
          end do
        end do
      else ! .not.enforce_stell_sym

        ii = 1
        do ll = 0, llrad
          id = Ate(lvol, 0, ii)%i(ll)
          if (id /= 0) then
            Ate(lvol, ideriv, ii)%s(ll) = solution(id)
          else
            Ate(lvol, ideriv, ii)%s(ll) = 0.0_rkind
          end if
          id = Aze(lvol, 0, ii)%i(ll)
          if (id /= 0) then
            Aze(lvol, ideriv, ii)%s(ll) = solution(id)
          else
            Aze(lvol, ideriv, ii)%s(ll) = 0.0_rkind
          end if
          Ato(lvol, ideriv, ii)%s(ll) = 0.0_rkind ! sin( m \t - n \z ) = 0 for (m,n)=(0,0)
          Azo(lvol, ideriv, ii)%s(ll) = 0.0_rkind ! sin( m \t - n \z ) = 0 for (m,n)=(0,0)
        end do
        do ii = 2, mn
          do ll = 0, llrad
            id = Ate(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Ate(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Ate(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
            id = Aze(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Aze(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Aze(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
            id = Ato(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Ato(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Ato(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
            id = Azo(lvol, 0, ii)%i(ll)
            if (id /= 0) then
              Azo(lvol, ideriv, ii)%s(ll) = solution(id)
            else
              Azo(lvol, ideriv, ii)%s(ll) = 0.0_rkind
            end if
          end do
        end do
      end if

    case ('P')

      !   FATAL( packab, .true., a trivial revision of packab is required to remove the if from the loop )

      solution = 0.0_rkind

      do ii = 1, mn
        do ll = 0, llrad
          id = Ate(lvol, 0, ii)%i(ll)
          if (id /= 0) solution(id) = Ate(lvol, ideriv, ii)%s(ll)
          id = Aze(lvol, 0, ii)%i(ll)
          if (id /= 0) solution(id) = Aze(lvol, ideriv, ii)%s(ll)
          if (ii > 1 .and. .not.enforce_stell_sym) then
            id = Ato(lvol, 0, ii)%i(ll)
            if (id /= 0) solution(id) = Ato(lvol, ideriv, ii)%s(ll)
            id = Azo(lvol, 0, ii)%i(ll)
            if (id /= 0) solution(id) = Azo(lvol, ideriv, ii)%s(ll)
          end if
        end do
      end do

    end select

  end subroutine packab

  subroutine packxi(NGdof, position, Mvol, mn, Nt, Nz, iRbc, iZbs, iRbs, iZbc, packorunpack, LComputeDerivatives, LComputeAxis)
    logical, intent(in) :: LComputeDerivatives ! indicates whether derivatives are to be calculated
    logical, intent(in) :: LComputeAxis ! if to recompute the axis

    integer, intent(in) :: NGdof, Mvol, mn, Nt, Nz
    real(rkind) :: position(0:NGdof), iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol)
    character :: packorunpack

    integer :: lvol, jj, irz, issym, idof, ivol

    idof = 0 ! initialize counter

    do lvol = 1, Mvol - 1 ! loop over internal interfaces
      do jj = 1, mn ! loop over Fourier harmonics
        do irz = 0, 1 ! loop over R & Z
          if (Igeometry < 3 .and. irz == 1) cycle ! no dependence on Z
          do issym = 0, 1 ! loop over even & odd
            if (enforce_stell_sym .and. issym == 1) cycle

            if (issym == 0 .and. irz == 1 .and. jj == 1) cycle ! no dependence on Zbs_{0,0}
            if (issym == 1 .and. irz == 0 .and. jj == 1) cycle ! no dependence on Rbs_{0,0}

            idof = idof + 1

            select case (packorunpack)

            case ('P') !   pack vector of unknowns
              if (irz == 0 .and. issym == 0) position(idof) = iRbc(jj, lvol) / psifactor(jj, lvol)
              if (irz == 1 .and. issym == 0) position(idof) = iZbs(jj, lvol) / psifactor(jj, lvol)
              if (irz == 0 .and. issym == 1) position(idof) = iRbs(jj, lvol) / psifactor(jj, lvol)
              if (irz == 1 .and. issym == 1) position(idof) = iZbc(jj, lvol) / psifactor(jj, lvol)

            case ('U') ! unpack vector of unknowns
              if (irz == 0 .and. issym == 0) iRbc(jj, lvol) = position(idof) * psifactor(jj, lvol)
              if (irz == 1 .and. issym == 0) iZbs(jj, lvol) = position(idof) * psifactor(jj, lvol)
              if (irz == 0 .and. issym == 1) iRbs(jj, lvol) = position(idof) * psifactor(jj, lvol)
              if (irz == 1 .and. issym == 1) iZbc(jj, lvol) = position(idof) * psifactor(jj, lvol)
            end select
          end do
        end do
      end do
    end do

    if (enforce_stell_sym) then ! iRbc(    ,0:Mvol) = 0.0_rkind
      iZbs(1, 0:Mvol) = 0.0_rkind
      iRbs(1:mn, 0:Mvol) = 0.0_rkind
      iZbc(1:mn, 0:Mvol) = 0.0_rkind
    else ! iRbc(    ,0:Mvol) = 0.0_rkind
      iZbs(1, 0:Mvol) = 0.0_rkind
      iRbs(1, 0:Mvol) = 0.0_rkind
      ! iZbc(    ,0:Mvol) = 0.0_rkind
    end if

    select case (packorunpack)
    case ('P')

    case ('U')
      ivol = 1 ! take care with ivol: this variable name might be a global variable, but here it is local

      if (LComputeAxis) then
        call rzaxis(Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), ivol,&
            & LComputeDerivatives)! set coordinate axis
      end if
    end select
  end subroutine packxi
end module packing_mod
