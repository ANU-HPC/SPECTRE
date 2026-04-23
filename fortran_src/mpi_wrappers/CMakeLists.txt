# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

create_global_source_file_list(mpi_wrapper_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    mpi_collectives_mod.F90
)

add_fortran_library(mpi_wrappers OBJECT ${mpi_wrapper_srcs})

target_link_libraries(mpi_wrappers PUBLIC
    common_compilation_options

    MPI::MPI_Fortran

    utils
)