# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

include(CMakeTools)
include(FortranTools)

# --- FFT backend selection (see ../CMakeLists.txt SPECTRE_FFT_BACKEND) ---------
# fft_backend_mod is provided by exactly one implementation file: the pure
# Fortran FFTW3-API one (backends fftw and mkl, which differ only in where
# fftw3.f03 and the library are resolved from, wired up below) or the PocketFFT
# one (which calls the small C++ bridge in extern/fft_shim).
if(SPECTRE_FFT_BACKEND STREQUAL "pocketfft")
  set(SPECTRE_FFT_BACKEND_SRC fft_backend_pocketfft_mod.F90)
else()
  set(SPECTRE_FFT_BACKEND_SRC fft_backend_fftw_mod.F90)
endif()

create_global_source_file_list(root_f90_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    allglobal_mod.F90
    base_functions_mod.F90
    beltrami_field_mod.F90
    beltrami_solver_mod.F90
    chebyshev_mod.F90
    constants_mod.F90
    coordinate_mod.F90
    cputiming_mod.F90
    diagnostics_mod.F90
    ${SPECTRE_FFT_BACKEND_SRC}
    forces_mod.F90
    freeb_mod.F90
    geometry_mod.F90
    hessian_mod.F90
    input_list_mod.F90
    input_parameters_mod.F90
    intghs_mod.F90
    magnetic_field_mod.F90
    matrices_mod.F90
    memory_mod.F90
    metric_mod.F90
    numerical_mod.F90
    packing_mod.F90
    perturbed_workspace_mod.F90
    plasma_current_mod.F90
    poincare_mod.F90
    preset_mod.F90
    profiles_distribution_mod.F90
    simulation_mod.F90
    sphdf5_mod.F90
    timer_type_mod.F90
    vector_potential_writer_mod.F90
    volume_mod.F90
    wrapper_funcs_mod.F90
    field_mod.F90
)

create_global_source_file_list(root_f_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    d1mach.f
    dcuhre.f
    i1mach.f
    iqpack.f
    minpack.f
    rksuite.f
)

# Differentiate because fixed-form Fortran files need different options
add_library(spec_f OBJECT ${root_f_srcs})
target_compile_options(spec_f
    PRIVATE
        "-cpp"
        $<$<Fortran_COMPILER_ID:GNU>:-ffree-line-length-none>
	$<$<AND:$<Fortran_COMPILER_ID:IntelLLVM>,$<CONFIG:Debug>>:-O0>
)

target_link_libraries(spec_f PUBLIC
    common_compilation_options

    ${OPENMP_LIB}
)

add_library(spec_f90 ${root_f90_srcs})

add_library(common_compilation_options INTERFACE)
target_compile_options(common_compilation_options
    INTERFACE
        $<$<Fortran_COMPILER_ID:GNU>:-ffree-line-length-none>
	$<$<AND:$<Fortran_COMPILER_ID:IntelLLVM>,$<CONFIG:Debug>>:-O0>
)

if(NOT USE_MPI_F08)
  target_compile_definitions(spec_f90 PRIVATE -DNO_MPI_F08)
endif()

# --- FFT backend library wiring (see the source selection at the top) ----------
if(SPECTRE_FFT_BACKEND STREQUAL "fftw")
  # FFTW::Double carries both libfftw3 and the include dir with fftw3.f03.
  set(SPECTRE_FFT_LIB FFTW::Double)
elseif(SPECTRE_FFT_BACKEND STREQUAL "mkl")
  # Intel oneMKL's drop-in FFTW3 interface: same Fortran source as fftw, but
  # fftw3.f03 and the library resolve to MKL. NO_DEFAULT_PATH: only MKL's own
  # copy of fftw3.f03 is acceptable -- the system search would find GPL FFTW's
  # in /usr/include first. Override with -DMKL_FFTW_INC=<dir> if MKL lives
  # elsewhere.
  find_path(MKL_FFTW_INC fftw3.f03
    PATHS /usr/include/mkl/fftw $ENV{MKLROOT}/include/fftw
    NO_DEFAULT_PATH
    DOC "Directory containing MKL's FFTW3 interface (fftw3.f03)")
  find_library(MKL_RT_LIB mkl_rt
    PATHS /usr/lib/x86_64-linux-gnu $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib
    DOC "Intel MKL single dynamic runtime library (mkl_rt)")
  if(NOT MKL_FFTW_INC OR NOT MKL_RT_LIB)
    message(FATAL_ERROR "SPECTRE_FFT_BACKEND=mkl: could not locate MKL's fftw3.f03 "
      "(MKL_FFTW_INC=${MKL_FFTW_INC}) and/or libmkl_rt (MKL_RT_LIB=${MKL_RT_LIB}). "
      "Install Intel MKL or set MKLROOT.")
  endif()
  message(STATUS "MKL FFTW3 interface: ${MKL_FFTW_INC}; lib: ${MKL_RT_LIB}")
  target_include_directories(spec_f90 PRIVATE ${MKL_FFTW_INC})
  set(SPECTRE_FFT_LIB ${MKL_RT_LIB})
elseif(SPECTRE_FFT_BACKEND STREQUAL "pocketfft")
  # PocketFFT (BSD-3-Clause, M. Reinecke): the single header pocketfft_hdronly.h,
  # fetched at configure time pinned to an exact upstream commit + SHA256 (not
  # vendored in-tree). For offline/air-gapped builds, set
  # -DFETCHCONTENT_SOURCE_DIR_POCKETFFT=<dir with pocketfft_hdronly.h> to use a
  # local copy instead of downloading.
  include(FetchContent)
  FetchContent_Declare(pocketfft
    URL https://raw.githubusercontent.com/mreineck/pocketfft/5f27d5a8f51c5c25030cb22abf434decc9faf0ff/pocketfft_hdronly.h
    URL_HASH SHA256=2e88a5b0addff3863a58258cc5bd92469a81c88cc4ec70cfc135e43f0cff8c14
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(pocketfft)
  add_library(spectre_fft STATIC "${CMAKE_SOURCE_DIR}/extern/fft_shim/spectre_fft2d_pocketfft.cpp")
  target_include_directories(spectre_fft PRIVATE ${pocketfft_SOURCE_DIR})
  set_target_properties(spectre_fft PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
  set(SPECTRE_FFT_LIB spectre_fft)
endif()
# ------------------------------------------------------------------------------

add_subdirectory(utils)
add_subdirectory(derived_types)
add_subdirectory(mpi_wrappers)
add_subdirectory(memory_management)
add_subdirectory(hdf5_wrappers)

target_link_libraries(spec_f90 PUBLIC
    common_compilation_options

    MPI::MPI_Fortran
    HDF5::HDF5
    ${SPECTRE_FFT_LIB}
    LAPACK::LAPACK
    ${OPENMP_LIB}

    spec_f
    hdf5_wrappers
    utils
    mpi_wrappers
    memory_management
    derived_types
)

if (NOT SKBUILD)
    add_executable(xspec xspech.F90)
    target_link_libraries(xspec PRIVATE spec_f90)
endif()
