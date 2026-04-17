create_global_source_file_list(derived_types_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    typedefns_mod.F90
)

add_fortran_library(derived_types OBJECT ${derived_types_srcs})

target_link_libraries(derived_types PUBLIC 
    common_compilation_options
    utils
)
