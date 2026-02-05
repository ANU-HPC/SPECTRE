__all__ = ["SPECTRE", "SPECTREout", "SPECNamelist"]

from importlib.util import find_spec

# main class for accessing the wrapper
from spectre.core import SPECTRE

# main class for force optimization
from spectre.force_minimizer import Force_Minimizer as Force_Minimizer

# main class for postprocessing
from spectre.postprocess.main import SPECTREout
from spectre.postprocess.namelist import SPECNamelist

# force objectives and some jacobians
from spectre.force_targets import (
    force_modes_dforce as force_modes_dforce,
    force_modes_dforce_with_grad as force_modes_dforce_with_grad,
    force_real as force_real,
    force_modes as force_modes,
    force_modes_targetmn as force_modes_targetmn,
    force_real_jac as force_real_jac,
    force_modes_jac as force_modes_jac,
    force_modes_speclike as force_modes_speclike,
    force_modes_jac_speclike as force_modes_jac_speclike,
)

from spectre.angle_targets import (
    angular_force_real as angular_force_real,
    angular_force_real_jac as angular_force_real_jac,
    angular_force_likespec as angular_force_likespec,
    angular_force_jac_likespec as angular_force_jac_likespec,
    len_cons_real as len_cons_real,
    len_cons_real_jac as len_cons_real_jac,
    eval_spectral_width as eval_spectral_width,
    eval_spectral_width_jac as eval_spectral_width_jac,
    get_theta_chord_length as get_theta_chord_length,
    get_theta_chord_length_jac as get_theta_chord_length_jac,
)

from spectre.input_dict import input_dict as input_dict

from spectre.isect_tester import check_intersection_wrapper as check_intersection_wrapper

from spectre.least_squares.least_squares import least_squares as least_squares

from spectre.misc import (
    Timer as Timer,
    find_res_infaces as find_res_infaces,
    get_xinit_specwrap as get_xinit_specwrap,
    create_runfile as create_runfile,
    get_fsa_wrapper as get_fsa_wrapper,
    get_freal_fsa as get_freal_fsa,
    get_freal_fsa_metric as get_freal_fsa_metric,
    get_xmn as get_xmn,
    reshape_fjac as reshape_fjac,
    reshape_modes_fjac as reshape_modes_fjac,
    get_fmn_force_mask as get_fmn_force_mask,
    remove_file as remove_file,
    infaces_unflatten as infaces_unflatten,
    infaces_flatten as infaces_flatten,
    unpack_interfaces as unpack_interfaces,
    pack_interfaces as pack_interfaces,
    check_fsa_vols as check_fsa_vols,
    get_vec_pot_flat as get_vec_pot_flat,
    get_vec_pot as get_vec_pot,
    update_bnorm as update_bnorm,
    filter_stellsym_pts as filter_stellsym_pts,
)

from .error_checking import (
    get_beltrami_errors as get_beltrami_errors,
    print_all_errors as print_all_errors,
)

from spectre.geometry import (
    get_coord_transform as get_coord_transform,
    get_jacobian as get_jacobian,
    get_interface_geometry as get_interface_geometry,
    get_coord_deriv as get_coord_deriv,
    get_rz_drzdxin as get_rz_drzdxin,
)

from spectre.plotting import (
    plot_fmn as plot_fmn,
    plot_fmn_anim as plot_fmn_anim,
    plot_infaces as plot_infaces,
    plot_infaces_helper as plot_infaces_helper,
    plot_jacobian as plot_jacobian,
    plot_grid as plot_grid,
    plot_true_mesh as plot_true_mesh,
)

from spectre.spec_fft import spec_fft as spec_fft, spec_fft_fast as spec_fft_fast

from spectre.numdiff_scipy import approx_derivative as approx_derivative

from spectre.cyl_island_width import get_cyl_island_width as get_cyl_island_width

from spectre.runners import run_local as run_local, run_slurm as run_slurm

from spectre.process_pool import run_pool as run_pool

from spectre.find_axis import find_axis_from_pncare as find_axis_from_pncare

from spectre.field_tracer import Field_Tracer as Field_Tracer

# Check if 'map2disc' is installed, import relevant functions if yes
has_map2disc = find_spec("map2disc") is not None

if has_map2disc:
    from spectre.mesh_gen import (
        get_inface_func as get_inface_func,
        gen_mesh_calc_rz as gen_mesh_calc_rz,
        gen_mesh_fft_rz as gen_mesh_fft_rz,
        gen_mesh_set_remn as gen_mesh_set_remn,
        gen_mesh_external as gen_mesh_external,
        gen_mesh_calc_deriv2 as gen_mesh_calc_deriv2,
        gen_infaces as gen_infaces,
    )

