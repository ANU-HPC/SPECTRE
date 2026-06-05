from spectre import SPECTRE, simsvmec2spectre
from spectre.converters import fixedbound_to_freeboundary, power_series, plot_freeboundary_modes, params_to_simsopt_boundary
from spectre.file_io import write_input_parameters_to_toml, read_input_parameters_from_toml
from simsopt.mhd import Vmec
from simsopt.geo import QfmResidual, QfmSurface, ToroidalFlux
from simsopt.field import BiotSavart, load_coils_from_makegrid_file
import logging
from simsopt.configs import get_data
import numpy as np
from matplotlib import pyplot as plt

#logging.basicConfig(level=logging.DEBUG)
plt.ion()  #interactive on, shows plots immediately

# Use the VMEC input file generated from the HINT-VMEC bridge at Proxima (from Camilla)
vmec = Vmec('./wout_li383_low_res_reference.nc')

#pressure = power_series(vmec.wout.am)  # evaluate the pressure
#s = np.linspace(0,1)
#plt.plot(s, pressure(s))
MPOL=8
NTOR=7

tfluxes = np.linspace(0,0.90, 7, endpoint=True)[1:]**.8  # tfluxes, but the last one is not exactly at the boundary. 
#generate concf. 
spectre_parameters = simsvmec2spectre(vmec, tfluxes=tfluxes, mpol=MPOL, ntor=NTOR, profile_type='current', plot=True, current_mode='sheet', interface_guess='wout', filename='spectre_fixb.toml') #write_input_parameters_to_toml(spectre_fixb, 'spectre_fb_med.toml')
plt.savefig("profiles.png")

# converge this separately, then read it
#spectre_parameters = read_input_parameters_from_toml('spectre_fb_opt.toml')
#new_phys = spectre_parameters.physics.model_copy(update={"mpol":18, "ntor":16})  #increase mpol and ntor, but start with what you have
#spectre_parameters = spectre_parameters.model_copy(update={"physics": new_phys})

# Read the computational boundary from the vacuum case 
boundary = params_to_simsopt_boundary(read_input_parameters_from_toml('ncsx2.toml'), which="comp").copy(mpol=MPOL, ntor=NTOR) # low number of QFM iterations


# generate coils from the coil file
_, _, _, _, bs = get_data('ncsx')
coils = bs.coils
# create a simsopt.field.biotsavart object: 
#bs = BiotSavart(coils)



spectre_fb_parameters = fixedbound_to_freeboundary(spectre_parameters, bs, boundary, vacuum=False)

new_phys = spectre_fb_parameters.physics.model_copy(update={"lbnszero": False, "lconstraint":3})
new_num = spectre_fb_parameters.numeric.model_copy(update={"lautoinitbn":True})
new_global = spectre_fb_parameters.global_.model_copy(update={"gbnbld":1.00})  #do NOT update boundary field because currently broken
 
spectre_fb_parameters = spectre_fb_parameters.model_copy(update={"physics": new_phys, "numeric":new_num, "global_":new_global})

plot_freeboundary_modes(bs, spectre_fb_parameters, log_scale=False)
plt.savefig("freeboundary_coeffs.png")

write_input_parameters_to_toml(spectre_fb_parameters, 'init.toml')


spectre_fb = SPECTRE.from_input_params(spectre_fb_parameters)
spectre_fb.create_spectreout().plot_infaces(zetan=0.5)





