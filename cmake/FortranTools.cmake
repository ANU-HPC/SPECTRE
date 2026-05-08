# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------
# Function: add_fortran_library
#
# Description:
#   Adds a Fortran library target and configures its module output directory.
#   Also sets up the module directory to be visible to other targets that link
#   against this library.
#
# Usage:
#   add_fortran_library(<target_name> <source1> [<source2> ...])
#
# Parameters:
#   <target_name> - The name of the Fortran library to be created.
#   <source1> ... - One or more Fortran source files to compile into the library.
#
# Behavior:
#   - Defines a library target named <target_name>.
#   - Sets the Fortran module output directory to <BINARY_DIR>/mod.
#   - Exposes the module directory via INTERFACE include paths for dependent targets.
#
# Example:
#   add_fortran_library(my_fortran_lib src/a.f90 src/b.f90)
#
# ----------------------------------------------------------------------------
function(add_fortran_library LIB)
    add_library(${LIB} ${ARGN})

    # set module path to LIB_DIR/mod
    get_target_property(LIB_DIR ${LIB} BINARY_DIR)
    set_target_properties(${LIB} PROPERTIES Fortran_MODULE_DIRECTORY ${LIB_DIR}/mod)

    # making LIB_DIR/mod available for libraries linking LIB
    target_include_directories(${LIB} INTERFACE ${LIB_DIR}/mod)
endfunction(add_fortran_library)