# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

include(CMakeTools)
include(FortranTools)

create_global_source_file_list(root_f90_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    allglobal_mod.F90
    base_functions_mod.F90
    beltrami_field_mod.F90
    beltrami_operator_mod.F90
    beltrami_solver_mod.F90
    beltrami_sparse_mod.F90
    chebyshev_mod.F90
    constants_mod.F90
    coordinate_mod.F90
    cputiming_mod.F90
    diagnostics_mod.F90
    fftw_interface_mod.F90
    forces_mod.F90
    freeb_mod.F90
    geometry_mod.F90
    gmres_mod.F90
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
    vc_matrix_mod.F90
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


add_subdirectory(utils)
add_subdirectory(derived_types)
add_subdirectory(mpi_wrappers)
add_subdirectory(memory_management)
add_subdirectory(hdf5_wrappers)

target_link_libraries(spec_f90 PUBLIC
    common_compilation_options

    MPI::MPI_Fortran
    HDF5::HDF5
    FFTW::Double
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
