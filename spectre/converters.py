import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from spectre.input_dict import input_dict
from spectre.file_io.toml_io import input_parameters_from_dict, write_input_parameters_to_toml

# flake8: noqa: E731


def vmec2spectre(
    vmec_obj, tfluxes=None, nvol=None, plot=False, mpol=6, ntor=5, current_mode="volume", filename="gen.toml"
):
    """Generates a SPECTRE input .toml file using  VMEC 'input.*' or 'wout_*' file. Sets up pressure prof, iota/current profile, and the plasma boundary. For the current-based case, it's possible to use either Ivolume integrated volume currents in spectre (current_mode='volume') or Isurf sheet currents (current_mode='sheet'). The profiles are normalized such that the total current matches VMEC."""

    # Get VMEC values
    if hasattr(vmec_obj, "indata"):  # from input file
        curtor = vmec_obj.indata.curtor
        phiedge = vmec_obj.indata.phiedge
        am = vmec_obj.indata.am
        ai = vmec_obj.indata.ai
        ac = vmec_obj.indata.ac
        nfp = vmec_obj.indata.nfp

        pl_boundary = vmec_obj.boundary

        pcurr_type = vmec_obj.indata.pcurr_type.decode()

        if vmec_obj.indata.ncurr == 0:
            profile = "iota"
        elif vmec_obj.indata.ncurr == 1:
            profile = "current"  # integrated current profile

    elif hasattr(vmec_obj, "wout"):  # from wout file
        phiedge = vmec_obj.wout.phi[-1]
        curtor = vmec_obj.wout.ctor
        am = vmec_obj.wout.am
        ai = vmec_obj.wout.ai
        ac = vmec_obj.wout.ac
        nfp = vmec_obj.wout.nfp

        pl_boundary = vmec_obj.boundary

        pcurr_type = "".join([s.decode() for s in vmec_obj.wout.pcurr_type]).strip()

        if np.max(np.abs(vmec_obj.wout.ai)) < 1e-8:
            profile = "current"
        else:
            profile = "iota"

    else:
        raise RuntimeError("The vmec object must have either indata or wout attribute!")

    # Get the plasma boundary
    num_modes = len(pl_boundary.m) // 2 + 1
    im_ = pl_boundary.m[:num_modes]
    in_ = pl_boundary.n[:num_modes]

    rbc = {}
    zbs = {}
    for ind in range(len(im_)):
        m = im_[ind]
        n = in_[ind]
        rbc[f"({m}, {n})"] = pl_boundary.get_rc(m, n)
        zbs[f"({m}, {n})"] = pl_boundary.get_zs(m, n)

    if pcurr_type != "power_series":
        raise RuntimeError(
            "Only 'power series' profile implemented, corresponding to j_phi(s) = I'(s) = Sum[ ac(j) s ** j] - Default"
        )

    if tfluxes is None and nvol is None:
        raise RuntimeError("Have to provide tfluxes or nvol")
    elif tfluxes is None:
        tfluxes = np.linspace(0, 1.0, nvol + 1, endpoint=True)[1:]
    else:
        nvol = len(tfluxes)

    tfluxes_ext = np.concatenate([[0.0], tfluxes])

    mu0 = 4 * np.pi * 1e-7

    # Set up the pressure profile
    vmec_pres_func = lambda s: np.sum([am[i] * s**i for i in range(len(am))], axis=0)
    profile_pres = np.zeros(nvol)  # per volume
    for i in range(nvol):
        a = tfluxes_ext[i]
        b = tfluxes_ext[i + 1]
        integ = quad(vmec_pres_func, a, b)[0]
        profile_pres[i] = mu0 * integ / (b - a)

    # Set up the other profile. Can be 'iota' or 'current'
    if profile == "iota":
        vmec_iota_func = lambda s: np.sum([ai[i] * s**i for i in range(len(ai))], axis=0)

        # Rot. transform on each interface. IOTA=OITA in SPECTRE
        profile_iota = vmec_iota_func(tfluxes)

    elif profile == "current":
        curr_s = lambda s: np.sum([ac[i] * s**i for i in range(len(ac))], axis=0)

        if current_mode == "volume":
            profile_volcurr = np.zeros(nvol)
            profile_surfcurr = np.zeros(nvol)

            for i in range(nvol):
                if i == 0:
                    a = 0
                    b = tfluxes[i]
                elif i == (nvol - 1):
                    a = tfluxes[i - 1]
                    b = 1.0
                else:
                    a = tfluxes[i - 1]
                    b = tfluxes[i]

                integ = quad(curr_s, a, b)[0]
                profile_volcurr[i] = integ  # / (b-a)

            for i in range(1, len(profile_volcurr)):
                profile_volcurr[i] += profile_volcurr[i - 1]

            profile_volcurr *= -curtor

            profile_volcurr *= mu0

        elif current_mode == "sheet":
            profile_volcurr = np.zeros(nvol)
            profile_surfcurr = np.zeros(nvol)  # per interface + extra zero (for some reason)

            # # Integrate VMEC j_phi in volume around an interface
            # for i in range(nvol-1):
            # 	if i == 0:
            # 		a = 0.
            # 	else:
            # 		a = (tfluxes[i] + tfluxes[i-1])/2.
            # 	if i == (nvol-0):
            # 		b = 1.
            # 	else:
            # 		b = (tfluxes[i] + tfluxes[i+1])/2.
            # 	integ = quad(curr_s, a, b)[0]
            # 	profile_surfcurr[i] = integ / (b-a)

            # Simplify evaluate VMEC j_phi on interface tfluxes
            profile_surfcurr[:-1] = curr_s(tfluxes[:-1])

            # Normalize the profile such that total current matches VMEC
            total_current_spec = np.sum(profile_surfcurr)  # / mu0
            if total_current_spec != 0.0:
                profile_surfcurr *= curtor / total_current_spec

            profile_surfcurr *= -1.0

            profile_surfcurr *= mu0
    else:
        raise RuntimeError("Profile has to be current or iota")

    # Create a toml file from the profiles

    dict_physics = input_dict()
    dict_numeric = input_dict()

    dict_physics.nfp = nfp
    dict_physics.nvol = nvol
    dict_physics.mpol = mpol
    dict_physics.ntor = ntor
    dict_physics.phiedge = phiedge
    dict_physics.tflux = tfluxes
    dict_physics.pressure = profile_pres
    dict_physics.pscale = 1.0
    dict_physics.lrad = np.array([mpol + 4] * nvol)
    if profile == "iota":
        dict_physics.lconstraint = 1
        dict_physics.iota = np.concatenate([[0.0], profile_iota])
        dict_physics.oita = dict_physics.iota
    elif profile == "current":
        dict_physics.lconstraint = 3
        dict_physics.isurf = profile_surfcurr
        dict_physics.ivolume = profile_volcurr
    dict_numeric.linitialize = 1  # Linear interpolation of RZ_mn
    dict_physics.rbc = rbc
    dict_physics.zbs = zbs

    dict_total = dict(physics=dict_physics, numeric=dict_numeric)
    dict_pydantic = input_parameters_from_dict(dict_total)
    write_input_parameters_to_toml(dict_pydantic, filename)

    if plot:
        stest = np.linspace(0, 1.0, 56, endpoint=True)  # s is normalized toroidal flux

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        prestest = vmec_pres_func(stest)
        ax[0].plot(stest, prestest)
        ax[0].set_title("Pressure")
        ax[0].hlines(profile_pres / mu0, tfluxes_ext[1:], tfluxes_ext[:-1], "red", lw=2)
        if profile == "iota":
            iotatest = vmec_iota_func(stest)
            ax[1].plot(stest, iotatest)
            ax[1].set_title("Rotational transform")
            ax[1].plot(tfluxes, profile_iota, "ro")
        elif profile == "current":
            if current_mode == "volume":
                currtest = [quad(curr_s, 0, lim)[0] for lim in stest]
                ax[1].plot(stest, currtest)
                ax[1].set_title("Integrated volume current (normalized)")
                ax[1].plot(tfluxes, profile_volcurr / np.max(profile_volcurr), "ro")
            elif current_mode == "sheet":
                currtest = curr_s(stest)
                ax[1].plot(stest, currtest/ np.max(currtest))
                ax[1].set_title("Sheet current (normalized)")
                if curtor != 0.0 :
                    ax[1].plot(tfluxes[:-1],profile_surfcurr[:-1]/ (-curtor / total_current_spec) /np.max(currtest) / mu0, "ro")
                else:
                    ax[1].plot(tfluxes[:-1], curtor*profile_surfcurr[:-1], "ro")
        for a in ax:
            a.vlines(
                tfluxes, a.get_ylim()[0], a.get_ylim()[1], color="k", lw=0.9, label="Interfaces"
            )
            a.set_xlabel("s / tflux")
            a.autoscale(enable=None, axis="y", tight=True)
            a.autoscale(enable=None, axis="x", tight=False)
        ax[0].legend(["VMEC", "SPECTRE", "Interfaces"])
        plt.tight_layout(w_pad=2)

    return dict_pydantic
