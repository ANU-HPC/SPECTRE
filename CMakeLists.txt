# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20.0)

# Ensure out-of-source build
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please, call CMake in the
    following way `cmake -S . -B <build_dir>`, where `<build_dir>` is different
    from the root.")
endif()

project(
  "SPECTRE"
  LANGUAGES Fortran C CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Selectable FFT backend for the Fortran core (see fortran_src/fft_backend_mod.F90).
# The default is PocketFFT so a plain build is fully MIT-redistributable; opt in to
# FFTW (fine for non-commercial use) or MKL (commercial-OK, fast) for more speed.
#   pocketfft : PocketFFT, BSD-3-Clause   (default; fully MIT-redistributable)
#   fftw      : FFTW, GPL                 (opt-in; fine for non-commercial use)
#   pffft     : PFFFT double, FFTPACK/BSD (opt-in; 2/3/5-smooth sizes only)
#   mkl       : Intel MKL FFTW3 interface (opt-in; proprietary but redistributable,
#               commercial-use OK; external dependency, not bundled)
set(SPECTRE_FFT_BACKEND "pocketfft" CACHE STRING "FFT backend: pocketfft, fftw, pffft, or mkl")
set_property(CACHE SPECTRE_FFT_BACKEND PROPERTY STRINGS pocketfft pffft fftw mkl)
string(TOLOWER "${SPECTRE_FFT_BACKEND}" SPECTRE_FFT_BACKEND)
if(NOT SPECTRE_FFT_BACKEND MATCHES "^(pocketfft|pffft|fftw|mkl)$")
  message(FATAL_ERROR "SPECTRE_FFT_BACKEND must be one of: pocketfft, pffft, fftw, mkl (got '${SPECTRE_FFT_BACKEND}')")
endif()
message(STATUS "SPECTRE FFT backend: ${SPECTRE_FFT_BACKEND}")

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HDF5 COMPONENTS Fortran REQUIRED)
if(SPECTRE_FFT_BACKEND STREQUAL "fftw")
  find_package(FFTW REQUIRED)
endif()

option(USE_OPENMP "Use OpenMP parallelization" ON)
option(USE_MPI_F08 "Use mpi_f08 interface (recommended)" ${MPI_Fortran_HAVE_F08_MODULE})

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(OPENMP_LIB OpenMP::OpenMP_Fortran)
else()
  set(OPENMP_LIB "")
endif()

add_subdirectory(fortran_src)

if (SKBUILD)
  add_subdirectory(spectre)
endif()
