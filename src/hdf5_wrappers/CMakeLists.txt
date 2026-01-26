create_global_source_file_list(hdf5_wrappers_srcs "${CMAKE_CURRENT_SOURCE_DIR}"
    hdf5_wrapper_mod.F90
)

add_fortran_library(hdf5_wrappers OBJECT ${hdf5_wrappers_srcs})

target_link_libraries(hdf5_wrappers PUBLIC
    common_compilation_options
    HDF5::HDF5

    utils
)