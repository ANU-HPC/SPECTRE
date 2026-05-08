# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

create_global_source_file_list(memory_management_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    memory_allocation_mod.F90
)

add_fortran_library(memory_management OBJECT ${memory_management_srcs})

target_link_libraries(memory_management PUBLIC 
    common_compilation_options
    utils
    derived_types
)
