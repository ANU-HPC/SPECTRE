# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

function(check_python_package package)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import ${package}"
        RESULT_VARIABLE _${package}_status
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_${package}_status EQUAL 0)
        message(STATUS "Python package '${package}' found.")
    else()
        message(FATAL_ERROR "Python package ${package} is missing.")
    endif()
endfunction()