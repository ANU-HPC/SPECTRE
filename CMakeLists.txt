# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
# SPDX-FileCopyrightText: 2026 Proxima Fusion
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
  LANGUAGES Fortran C
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Licensing switch: default builds use the fastest backends regardless of their
# license (FFTW for the FFT, GPL; CyRK for the Python field-line tracer,
# CC-BY-NC-SA). With SPECTRE_COMMERCIAL_USE=ON the defaults switch to
# permissively-licensed alternatives (pocketfft FFT, scipy tracer).
option(SPECTRE_COMMERCIAL_USE "Default to permissively-licensed backends (pocketfft FFT, scipy tracer)" OFF)

# Selectable FFT backend for the Fortran core. The fft_backend_mod module has one
# Fortran implementation file per backend family (fortran_src/fft_backend_*_mod.F90);
# exactly one is compiled in (see fortran_src/CMakeLists.txt).
#   fftw      : FFTW, GPL                    (default; fastest)
#   mkl       : Intel oneMKL FFTW3 interface (proprietary but redistributable,
#               commercial-use OK; external dependency, not bundled)
#   pocketfft : PocketFFT, BSD-3-Clause      (default with SPECTRE_COMMERCIAL_USE;
#               fetched at configure time, not vendored)
set(SPECTRE_FFT_BACKEND "" CACHE STRING "FFT backend: fftw, mkl, or pocketfft (empty = automatic)")
set_property(CACHE SPECTRE_FFT_BACKEND PROPERTY STRINGS "" fftw mkl pocketfft)
string(TOLOWER "${SPECTRE_FFT_BACKEND}" SPECTRE_FFT_BACKEND)
if(SPECTRE_FFT_BACKEND STREQUAL "")
  if(SPECTRE_COMMERCIAL_USE)
    set(SPECTRE_FFT_BACKEND "pocketfft")
  else()
    set(SPECTRE_FFT_BACKEND "fftw")
  endif()
endif()
if(NOT SPECTRE_FFT_BACKEND MATCHES "^(fftw|mkl|pocketfft)$")
  message(FATAL_ERROR "SPECTRE_FFT_BACKEND must be one of: fftw, mkl, pocketfft (got '${SPECTRE_FFT_BACKEND}')")
endif()
if(SPECTRE_COMMERCIAL_USE AND SPECTRE_FFT_BACKEND STREQUAL "fftw")
  message(WARNING "SPECTRE_COMMERCIAL_USE=ON but SPECTRE_FFT_BACKEND=fftw: FFTW is GPL-licensed.")
endif()
message(STATUS "SPECTRE FFT backend: ${SPECTRE_FFT_BACKEND} (SPECTRE_COMMERCIAL_USE=${SPECTRE_COMMERCIAL_USE})")
if(SPECTRE_FFT_BACKEND STREQUAL "pocketfft")
  # Only the pocketfft bridge is C++; fftw/mkl builds are pure Fortran and need no
  # C++ compiler. Must be enabled here at top level so every directory links the
  # C++ runtime correctly.
  enable_language(CXX)
endif()

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
