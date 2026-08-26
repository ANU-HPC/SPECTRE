# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

__all__ = ["SPECTRE", "SPECTREout"]

from importlib.util import find_spec

from spectre.angle_targets import (
    angular_force_jac_likespec as angular_force_jac_likespec,
)
from spectre.angle_targets import (
    angular_force_likespec as angular_force_likespec,
)
from spectre.angle_targets import (
    angular_force_real as angular_force_real,
)
from spectre.angle_targets import (
    angular_force_real_jac as angular_force_real_jac,
)
from spectre.angle_targets import (
    eval_spectral_width as eval_spectral_width,
)
from spectre.angle_targets import (
    eval_spectral_width_jac as eval_spectral_width_jac,
)
from spectre.angle_targets import (
    get_theta_chord_length as get_theta_chord_length,
)
from spectre.angle_targets import (
    get_theta_chord_length_jac as get_theta_chord_length_jac,
)
from spectre.angle_targets import (
    len_cons_real as len_cons_real,
)
from spectre.angle_targets import (
    len_cons_real_jac as len_cons_real_jac,
)
from spectre.converters import simsvmec2spectre as simsvmec2spectre

# main class for accessing the wrapper
from spectre.core import SPECTRE
from spectre.cyl_island_width import get_cyl_island_width as get_cyl_island_width
from spectre.field_tracer import Field_Tracer as Field_Tracer
from spectre.fixed_point import find_axis as find_axis

# main class for force optimization
from spectre.force_minimizer import Force_Minimizer as Force_Minimizer
from spectre.force_targets import (
    force_modes as force_modes,
)

# force objectives and some jacobians
from spectre.force_targets import (
    force_modes_dforce as force_modes_dforce,
)
from spectre.force_targets import (
    force_modes_dforce_with_grad as force_modes_dforce_with_grad,
)
from spectre.force_targets import (
    force_modes_jac as force_modes_jac,
)
from spectre.force_targets import (
    force_modes_jac_speclike as force_modes_jac_speclike,
)
from spectre.force_targets import (
    force_modes_speclike as force_modes_speclike,
)
from spectre.force_targets import (
    force_modes_targetmn as force_modes_targetmn,
)
from spectre.force_targets import (
    force_real as force_real,
)
from spectre.force_targets import (
    force_real_jac as force_real_jac,
)
from spectre.force_targets import (
    force_real_jvp as force_real_jvp,
)
from spectre.fourier import spec_fft as spec_fft
from spectre.fourier import spec_fft_fast as spec_fft_fast
from spectre.geometry import (
    get_coord_deriv as get_coord_deriv,
)
from spectre.geometry import (
    get_coord_transform as get_coord_transform,
)
from spectre.geometry import (
    get_interface_geometry as get_interface_geometry,
)
from spectre.geometry import (
    get_jacobian as get_jacobian,
)
from spectre.geometry import (
    get_rz_drzdxin as get_rz_drzdxin,
)
from spectre.input_dict import input_dict as input_dict

# different interface dof options
from spectre.interface_dof import (
    identity_dof as identity_dof,
)
from spectre.interface_dof import (
    reduced_rzmn_dof as reduced_rzmn_dof,
)
from spectre.isect_tester import check_intersection_wrapper as check_intersection_wrapper
from spectre.least_squares.least_squares import least_squares as least_squares
from spectre.numdiff_scipy import approx_derivative as approx_derivative
from spectre.plotting import (
    plot_beltrami_error as plot_beltrami_error,
)
from spectre.plotting import (
    plot_fmn as plot_fmn,
)
from spectre.plotting import (
    plot_fmn_anim as plot_fmn_anim,
)
from spectre.plotting import (
    plot_grid as plot_grid,
)
from spectre.plotting import (
    plot_infaces as plot_infaces,
)
from spectre.plotting import (
    plot_infaces_helper as plot_infaces_helper,
)
from spectre.plotting import (
    plot_jacobian as plot_jacobian,
)
from spectre.plotting import (
    plot_true_mesh as plot_true_mesh,
)
from spectre.plotting import (
    plot_worst_beltrami as plot_worst_beltrami,
)
from spectre.plotting import (
    plot_worst_jacobian as plot_worst_jacobian,
)

# main class for postprocessing
from spectre.postprocess.main import SPECTREout
from spectre.process_pool import run_pool as run_pool
from spectre.runners import run_local as run_local
from spectre.runners import run_slurm as run_slurm
from spectre.stability import calc_hessian as calc_hessian
from spectre.surfaces.surf import Surface as Surface
from spectre.surfaces.surf_henneberg import SurfaceHenneberg as SurfaceHenneberg
from spectre.surfaces.surf_rz import SurfaceRZFourier as SurfaceRZFourier
from spectre.utils import (
    Timer as Timer,
)
from spectre.utils import (
    check_fsa_vols as check_fsa_vols,
)
from spectre.utils import (
    create_runfile as create_runfile,
)
from spectre.utils import (
    filter_stellsym_pts as filter_stellsym_pts,
)
from spectre.utils import (
    find_res_infaces as find_res_infaces,
)
from spectre.utils import (
    get_fmn_force_mask as get_fmn_force_mask,
)
from spectre.utils import (
    get_freal_fsa as get_freal_fsa,
)
from spectre.utils import (
    get_freal_fsa_metric as get_freal_fsa_metric,
)
from spectre.utils import (
    get_fsa_wrapper as get_fsa_wrapper,
)
from spectre.utils import (
    get_vec_pot as get_vec_pot,
)
from spectre.utils import (
    get_vec_pot_flat as get_vec_pot_flat,
)
from spectre.utils import (
    get_xinit_specwrap as get_xinit_specwrap,
)
from spectre.utils import (
    get_xmn as get_xmn,
)
from spectre.utils import (
    infaces_flatten as infaces_flatten,
)
from spectre.utils import (
    infaces_unflatten as infaces_unflatten,
)
from spectre.utils import (
    pack_interfaces as pack_interfaces,
)
from spectre.utils import (
    reshape_fjac as reshape_fjac,
)
from spectre.utils import (
    reshape_modes_fjac as reshape_modes_fjac,
)
from spectre.utils import (
    save_errors_to_h5 as save_errors_to_h5,
)
from spectre.utils import (
    unpack_interfaces as unpack_interfaces,
)

from .error_checking import (
    beltrami_error_grid as beltrami_error_grid,
)
from .error_checking import (
    get_beltrami_errors as get_beltrami_errors,
)
from .error_checking import (
    print_all_errors as print_all_errors,
)

# Check if 'map2disc' is installed, import relevant functions if yes
has_map2disc = find_spec("map2disc") is not None

if has_map2disc:
    from spectre.mesh_gen import (
        gen_infaces as gen_infaces,
    )
    from spectre.mesh_gen import (
        gen_mesh_external as gen_mesh_external,
    )
