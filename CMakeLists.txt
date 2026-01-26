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

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HDF5 COMPONENTS Fortran REQUIRED)
find_package(FFTW REQUIRED)

option(USE_OPENMP "Use OpenMP parallelization" ON)
option(USE_MPI_F08 "Use mpi_f08 interface (recommended)" ${MPI_Fortran_HAVE_F08_MODULE})

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(OPENMP_LIB OpenMP::OpenMP_Fortran)
else()
  set(OPENMP_LIB "")
endif()

add_subdirectory(src)

if (SKBUILD)
  add_subdirectory(spectre)
endif()
