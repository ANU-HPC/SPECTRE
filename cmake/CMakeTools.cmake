# ---------------------------------------------------------------------------
# Function: create_global_source_file_list
#
# Description:
#   Takes a list of relative file paths and prepends a common root path to
#   each element, producing a new list of full paths. The result is stored
#   in the specified output variable and is global to the project.
#
# Usage:
#   create_global_source_file_list(<output_variable> <root_path> <file1> [<file2> ...])
#
# Parameters:
#   <output_variable> - Name of the variable to store the resulting list.
#   <root_path>       - Path to prepend to each file.
#   <file1> ...       - One or more relative file names.
#
# Example:
#   create_global_source_file_list(my_files "${CMAKE_CURRENT_SOURCE_DIR}"
#       "src/a.f90" "src/b.f90")
#
#   # Result:
#   #   my_files = "${CMAKE_CURRENT_SOURCE_DIR}/src/a.f90"
#   #              "${CMAKE_CURRENT_SOURCE_DIR}/src/b.f90"
# ---------------------------------------------------------------------------
function(create_global_source_file_list file_list root_path)
    set(result "")
    foreach(item IN LISTS ARGN)
        list(APPEND result "${root_path}/${item}")
    endforeach()

    set(${file_list} "${result}" CACHE INTERNAL "")
endfunction(create_global_source_file_list)