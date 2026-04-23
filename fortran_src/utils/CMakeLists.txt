# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

create_global_source_file_list(utils_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    conversion_mod.F90
    errors_mod.F90
    io_mod.F90
    precisions_mod.F90
)

add_fortran_library(utils OBJECT ${utils_srcs})

target_link_libraries(utils PUBLIC
    common_compilation_options

    MPI::MPI_Fortran
)
