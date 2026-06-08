# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

include(CMakeTools)
include(FortranTools)

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
    fft_backend_mod.F90
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

# --- FFT backend selection (see ../CMakeLists.txt SPECTRE_FFT_BACKEND) ---------
# fft_backend_mod.F90 is CPP-switched on FFT_BACKEND_{FFTW,POCKETFFT,PFFFT};
# pocketfft/pffft additionally need their vendored C/C++ shim in extern/.
# The "mkl" backend reuses the FFTW code path (Intel MKL provides the FFTW3 ABI).
set(SPECTRE_EXTERN "${CMAKE_SOURCE_DIR}/extern")
if(SPECTRE_FFT_BACKEND STREQUAL "fftw")
  target_compile_definitions(spec_f90 PRIVATE -DFFT_BACKEND_FFTW)
  set(SPECTRE_FFT_LIB FFTW::Double)
elseif(SPECTRE_FFT_BACKEND STREQUAL "mkl")
  # Intel MKL ships a drop-in FFTW3 interface; reuse the FFTW3-ABI Fortran path
  # but resolve fftw3.f03 and the implementation from MKL (no GPL FFTW).
  target_compile_definitions(spec_f90 PRIVATE -DFFT_BACKEND_FFTW)
  find_path(MKL_FFTW_INC fftw3.f03
    PATHS /usr/include/mkl/fftw $ENV{MKLROOT}/include/fftw $ENV{MKLROOT}/include
    DOC "Directory containing MKL's FFTW3 interface (fftw3.f03)")
  find_library(MKL_RT_LIB mkl_rt
    PATHS /usr/lib/x86_64-linux-gnu $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib
    DOC "Intel MKL single dynamic runtime library (mkl_rt)")
  if(NOT MKL_FFTW_INC OR NOT MKL_RT_LIB)
    message(FATAL_ERROR "SPECTRE_FFT_BACKEND=mkl: could not locate MKL's fftw3.f03 "
      "(MKL_FFTW_INC=${MKL_FFTW_INC}) and/or libmkl_rt (MKL_RT_LIB=${MKL_RT_LIB}). "
      "Install Intel MKL or set MKLROOT.")
  endif()
  target_include_directories(spec_f90 PRIVATE ${MKL_FFTW_INC})
  set(SPECTRE_FFT_LIB ${MKL_RT_LIB})
  message(STATUS "MKL FFTW3 interface: ${MKL_FFTW_INC}; lib: ${MKL_RT_LIB}")
elseif(SPECTRE_FFT_BACKEND STREQUAL "pocketfft")
  target_compile_definitions(spec_f90 PRIVATE -DFFT_BACKEND_POCKETFFT)
  add_library(spectre_fft STATIC ${SPECTRE_EXTERN}/fft_shim/spectre_fft2d_pocketfft.cpp)
  target_include_directories(spectre_fft PRIVATE
    ${SPECTRE_EXTERN}/fft_shim ${SPECTRE_EXTERN}/pocketfft)
  set_target_properties(spectre_fft PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
  set(SPECTRE_FFT_LIB spectre_fft)
elseif(SPECTRE_FFT_BACKEND STREQUAL "pffft")
  target_compile_definitions(spec_f90 PRIVATE -DFFT_BACKEND_PFFFT)
  add_library(spectre_fft STATIC
    ${SPECTRE_EXTERN}/fft_shim/spectre_fft2d_pffft.c
    ${SPECTRE_EXTERN}/pffft/src/pffft_double.c
    ${SPECTRE_EXTERN}/pffft/src/pffft_common.c)
  target_include_directories(spectre_fft PRIVATE
    ${SPECTRE_EXTERN}/fft_shim
    ${SPECTRE_EXTERN}/pffft/include/pffft
    ${SPECTRE_EXTERN}/pffft/src)
  set_target_properties(spectre_fft PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
