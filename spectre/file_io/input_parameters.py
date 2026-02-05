"""
Define the data models for SPEC input parameters using Pydantic.
The model uses serializers to generate human-readable files for writing with toml (in toml_io.py), 
and uses validators to cast incoming data into the format expected by SPEC, which are numpy arrays
of much larger sizes. 
"""


from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator, field_serializer, ConfigDict, field_validator, PrivateAttr
import numpy.typing as npt
import numpy as np

mnvol = 256
mmpol = 128
mntor = 128
mnmax = mntor + 1 + mmpol * (2 * mntor + 1)

# these arrays have length total plasma volumes (nvol)
arrays_of_length_nvol = ["pressure", "adiabatic", "mu", "isurf"]
# these arrays have length that depends on freeboundary (called mvol in SPEC)
arrays_of_length_mvol = ["lrad", "tflux", "ivolume", "pflux", "helicity", "pl", "ql", "pr", "qr", "lp", "lq", "rp", "rq"]
# some arrays are one element longer, because they are indexed strangely
arrays_of_length_mvol_plus = ["iota", "oita"]
# some arrays are only on interfaces (nvol-1)
arrays_of_length_nvol_minus_1 = []  # is oita not minus two in fixed boundary?
#  array sizes for fourier coeffs
arrays_fourier = ["rbc", "zbs", "rbs", "zbc", "rwc", "zws", "rws", "zwc", "vns", "bns", "vnc", "bnc"]


def _dict_to_specarray(mapping: Mapping[Any, Any]) -> np.ndarray:
    """
    convert the fourier coefficients given by a dictionary to elements of 
    dfthe SPEC array
    """

    arr = np.zeros((2 * mntor + 1, 2 * mmpol + 1), dtype=float)
    
    for raw_key, coeff in mapping.items():
        text = str(raw_key).strip()
        if not (text.startswith("(") and text.endswith(")")):
            raise ValueError(f"Cannot parse Fourier index '{raw_key}'.")
        parts = [part.strip() for part in text[1:-1].split(",")]
        if len(parts) != 2:
            raise ValueError(f"Cannot parse Fourier index '{raw_key}'.")
        m_idx, n_idx = int(parts[0]), int(parts[1])
        if abs(n_idx) > mntor:
            raise ValueError(""f"Toroidal index n={n_idx} exceeds configured maximum ±{mntor}.")
        if m_idx < 0:
            raise ValueError(f"Poloidal index m={m_idx} cannot be negative.")
        if m_idx > mmpol:
            raise ValueError(
                f"Poloidal index m={m_idx} exceeds configured maximum ±{mmpol}."
            )
        if m_idx == 0 and n_idx < 0:
            continue
        if not np.isfinite(coeff):
            raise ValueError(f"Fourier coefficient for mode ({raw_key}) is not finite.")

        arr[mntor + n_idx, mmpol + m_idx] = coeff
    return arr


class PhysicsParameters(BaseModel):
    igeometry: int = Field(default=3, ge=1, le=3)
    nfp: int = 1
    nvol: int = 1
    mpol: int = 0
    ntor: int = 0
    lconstraint: int = -1
    mupfits: int = 8
    ladiabatic: int = 0
    num_modes: int = Field(default=0, exclude=True)  # helper for reading interface guesses from file
    mupftol: float = 1.0e-14
    rpol: float = 1.0
    rtor: float = 1.0
    lreflect: int = 0
    lfreebound: int = 0
    phiedge: float = 1.0
    curtor: float = 0.0
    curpol: float = 0.0
    gamma: float = 0.0
    pscale: float = 0.0
    enforce_stell_sym: bool = True
    lrad: npt.NDArray[np.int32] = Field(default_factory=lambda: np.full(mnvol + 1, 4, dtype=int))
    pl: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    ql: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    pr: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    qr: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    lp: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    lq: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    rp: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    rq: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros(mnvol + 1, dtype=int))
    tflux: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    pflux: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    pressure: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    adiabatic: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    mu: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    ivolume: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    iota: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    oita: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol + 1))
    isurf: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol))
    helicity: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mnvol))
    rac: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mntor + 1))
    zas: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mntor + 1))
    ras: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mntor + 1))
    zac: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros(mntor + 1))
    rbc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    zbs: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    rbs: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    zbc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    rwc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    zws: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    rws: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    zwc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    vns: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    bns: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    vnc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    bnc: npt.NDArray[np.float64] = Field(default_factory=lambda: np.zeros((2 * mntor + 1, 2 * mmpol + 1)))
    allrzrz: npt.NDArray[np.float64] = None
    mmrzrz: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros((mnmax), dtype=np.int32), exclude=True)  # helper index array for mpol value in each line
    nnrzrz: npt.NDArray[np.int32] = Field(default_factory=lambda: np.zeros((mnmax), dtype=np.int32), exclude=True)  # helper index array for ntor value in each line - private so not serialized

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer(*arrays_of_length_nvol)
    def serialize_radial_arrays_nvol(self, value):
        """
        serialize the radial arrays that provide a value for each volume/interface internal to the configuration
        """
        arr = np.atleast_1d(np.asarray(value))
        return arr[:self.nvol].tolist()
    
    @field_serializer(*arrays_of_length_mvol)
    def serialize_radial_arrays_mvol(self, value):
        """
        serialize the arrays that are not always nvol long but depending on freeboundary
        can be one longer (are thus defined in the vacuum region enclosing the plasma).
        """
        array_length = self.nvol + 1 if self.lfreebound == 1 else self.nvol
        arr = np.atleast_1d(np.asarray(value))
        return arr[:array_length].tolist()
    
    @field_serializer(*arrays_of_length_mvol_plus)
    def serialize_radial_arrays_mvol_plus(self, value):
        """
        serialize the arrays that are not always nvol long but depending on freeboundary
        can be one longer (are thus defined in the vacuum region enclosing the plasma).
        """
        mvol = self.nvol + 1 if self.lfreebound == 1 else self.nvol
        array_length = mvol+1
        arr = np.atleast_1d(np.asarray(value))
        return arr[:array_length].tolist()
    
    @field_serializer(*arrays_of_length_nvol_minus_1)
    def serialize_radial_arrays_nvol_minus_1(self, value):
        """
        serialize the radial arrays that are only nvol-1 long
        """
        arr = np.atleast_1d(np.asarray(value))
        return arr[: self.nvol - 1].tolist()

    @field_serializer("rac", "zas", "ras", "zac")
    def serialize_toroidal_arrays(self, value):
        return value[:self.ntor+1].tolist()

    @field_serializer(*arrays_fourier)
    def serialize_fourier_arrays(self, value):
        nt = self.ntor
        mp = self.mpol
        coeffs = {}
        for m in range(0, mp + 1):
            for n in range(-nt, nt + 1):
                if m == 0 and n < 0:
                    continue
                p_idx = mmpol + m
                t_idx = mntor + n
                coeff = float(value[t_idx, p_idx]) # toroidal first becase FORTRAN
                if coeff != 0.0:
                    coeffs[f'({m}, {n})'] = coeff
        return coeffs

    @field_serializer("allrzrz")
    def serialize_guess(self, allrzrz):
        """
        The interface guess is stored in allrzrz with index helper arrays mmrzrz and nnrzrz.
        Our job is to reconstruct a human readable mapping from this.
        
        This helper array has the shape it has because it was dumped at the end of the namelist. 
        In the future this will be deprecated for direct writing of the interface guesses one by one. 
        """
        if allrzrz is None:
            return dict()
        nt = self.ntor
        mp = self.mpol
        interface_guess = {}
        for ivol in range(1, self.nvol+1):
            interface_guess[f'interface_{ivol}'] = {}
            rbc = {}
            zbs = {}
            rbs = {}
            zbc = {}
            interface_modes = allrzrz[:, ivol-1, :]
            for modelist_index in range(interface_modes.shape[1]): # run over modes:
                m_idx = self.mmrzrz[modelist_index]
                n_idx = self.nnrzrz[modelist_index]
                if m_idx > mp or abs(n_idx) > nt:
                    continue  # mode not represented in this config
                coeff_string = f'({m_idx}, {n_idx})'
                row_of_coeffs = interface_modes[:, modelist_index]
                rbc_coeff, zbs_coeff, rbs_coeff, zbc_coeff = float(row_of_coeffs[0]), float(row_of_coeffs[1]), float(row_of_coeffs[2]), float(row_of_coeffs[3])
                if rbc_coeff != 0.0:
                    rbc[coeff_string] = rbc_coeff
                if zbs_coeff != 0.0:
                    zbs[coeff_string] = zbs_coeff
                if rbs_coeff != 0.0:
                    rbs[coeff_string] = rbs_coeff
                if zbc_coeff != 0.0:
                    zbc[coeff_string] = zbc_coeff
            interface_guess[f'interface_{ivol}']['rbc'] = rbc
            interface_guess[f'interface_{ivol}']['zbs'] = zbs
            interface_guess[f'interface_{ivol}']['rbs'] = rbs
            interface_guess[f'interface_{ivol}']['zbc'] = zbc
        return interface_guess
    
    @model_validator(mode="before")
    @classmethod
    def set_guess(cls, values):
        """
        prepare the allrzrz array from the input mapping if provided. 
        don't judge me, this keeps the toml file with a clean, readable guess, and
        keeps the changes to the interface to the SPEC fortran side minimal. 
        In the future the input_list_mod will (hopefully) be replaced with a 
        interface_mod that passes all relevant information for the SPEC state
        to the FORTRAN memory, allowing switching between configurations without
        running all intialization steps...
        """
        if "allrzrz" in values:
            number_of_interfaces = values.get("nvol") 
            fortran_shape_of_allrzrz = [4, number_of_interfaces, mnmax]
            

            if isinstance(values['allrzrz'], np.ndarray) and values["allrzrz"].shape == fortran_shape_of_allrzrz:  # should not get here
                raise ValueError("allrzrz array has correct shape already, should not get here.")
                return values  # already in correct shape

            spec_allrzrz = np.zeros(fortran_shape_of_allrzrz, dtype=float)
            spec_mmrzrz = np.zeros((mnmax), dtype=int)
            spec_nnrzrz = np.zeros((mnmax), dtype=int)
            if isinstance(values["allrzrz"], Mapping):  # we read a toml file, we parse thusly:
                guess_mapping = values["allrzrz"]
                
                # Check if the mapping is empty or contains only empty interface dicts
                # If so, we should not create an allrzrz array (no guess provided)
                has_data = False
                for interface_key, interface_dict in guess_mapping.items():
                    if isinstance(interface_dict, Mapping):
                        for coeff_type in ("rbc", "zbs", "rbs", "zbc"):
                            if interface_dict.get(coeff_type):
                                has_data = True
                                break
                    if has_data:
                        break
                if not has_data:
                    del values["allrzrz"]
                    return values

                if not isinstance(guess_mapping, Mapping):
                    raise ValueError("allrzrz must be a mapping of interface names to fourier coeff dicts.")
                
                # First pass: collect all unique mode keys from all interfaces
                all_keys = set()
                for interface_idx in range(1, number_of_interfaces + 1):
                    interface_dict = guess_mapping.get(f"interface_{interface_idx}", {})
                    if isinstance(interface_dict, Mapping):
                        rbc_dict = interface_dict.get("rbc", {})
                        zbs_dict = interface_dict.get("zbs", {})
                        rbs_dict = interface_dict.get("rbs", {})
                        zbc_dict = interface_dict.get("zbc", {})
                        all_keys |= set(rbc_dict.keys()) | set(zbs_dict.keys()) | set(rbs_dict.keys()) | set(zbc_dict.keys())
                
                # Parse keys and create a sorted list for deterministic ordering
                def parse_key(raw_key):
                    text = str(raw_key).strip()
                    if not (text.startswith("(") and text.endswith(")")):
                        raise ValueError(f"Cannot parse Fourier index '{raw_key}'.")
                    parts = [part.strip() for part in text[1:-1].split(",")]
                    if len(parts) != 2:
                        raise ValueError(f"Cannot parse Fourier index '{raw_key}'.")
                    return int(parts[0]), int(parts[1])
                
                # Sort keys by (m, n) for deterministic ordering
                key_to_mn = {key: parse_key(key) for key in all_keys}
                sorted_keys = sorted(all_keys, key=lambda k: key_to_mn[k])
                
                # Second pass: populate arrays with consistent indexing
                for interface_idx in range(1, number_of_interfaces + 1):
                    interface_dict = guess_mapping.get(f"interface_{interface_idx}", {})
                    if not isinstance(interface_dict, Mapping):
                        raise ValueError(f"Interface guess for {interface_idx} must be a mapping of fourier coeff dicts.")
                    rbc_dict = interface_dict.get("rbc", {})
                    zbs_dict = interface_dict.get("zbs", {})
                    rbs_dict = interface_dict.get("rbs", {})
                    zbc_dict = interface_dict.get("zbc", {})
                    
                    for list_label, raw_key in enumerate(sorted_keys):
                        m_idx, n_idx = key_to_mn[raw_key]
                        rbc_coeff = rbc_dict.get(raw_key, 0.0)
                        zbs_coeff = zbs_dict.get(raw_key, 0.0)
                        rbs_coeff = rbs_dict.get(raw_key, 0.0)
                        zbc_coeff = zbc_dict.get(raw_key, 0.0)
                        # Use 1-based indexing for list_label to match Fortran convention (index 0 is unused)
                        fortran_idx = list_label + 1
                        spec_allrzrz[:, interface_idx-1, fortran_idx] = [rbc_coeff, zbs_coeff, rbs_coeff, zbc_coeff]
                        spec_mmrzrz[fortran_idx] = m_idx
                        spec_nnrzrz[fortran_idx] = n_idx
                values["allrzrz"] = spec_allrzrz
                values["mmrzrz"] = spec_mmrzrz
                values["nnrzrz"] = spec_nnrzrz
                values["num_modes"] = len(sorted_keys) + 1  # total number of modes (+1 for Fortran 1-based loop)
                return values
            elif isinstance(values["allrzrz"], np.ndarray): # Probably gotten from namelist when converting an .sp file
                print(f'Reformatting allrzrz array from namelist input. Shape is now: {values["allrzrz"].shape}')
                for row_idx, row in enumerate(values["allrzrz"]):
                    row_idx += 1 # fortran indexing
                    spec_mmrzrz[row_idx] = int(row[0])
                    spec_nnrzrz[row_idx] = int(row[1])
                    coeffs = np.asarray(row[2:], dtype=float)
                    available_sets = coeffs.size // 4
                    filled_interfaces = min(number_of_interfaces, available_sets)
                    if filled_interfaces:
                        chunk = coeffs[: filled_interfaces * 4].reshape(filled_interfaces, 4)
                        spec_allrzrz[:, :filled_interfaces, row_idx] = chunk.T
                values["allrzrz"] = spec_allrzrz
                values["mmrzrz"] = spec_mmrzrz
                values["nnrzrz"] = spec_nnrzrz
                values["num_modes"] = row_idx  # last
                return values
        return values  # unknown format, let other validators handle the error
        

    @field_validator("lrad", "pl", "ql", "pr", "qr", "lp", "lq", "rp", "rq", mode="before")
    def pad_radial_int_arrays(cls, value):
        """
        pad the 1D int arrays to the SPEC expected size
        """
        arr = np.zeros(mnvol+1, dtype=int)
        src = np.atleast_1d(np.asarray(value, dtype=int))
        arr[:src.size] = src[:arr.size]
        return arr
    
    @field_validator("adiabatic", "tflux", "pflux", "pressure", "mu",
                     "ivolume", "iota", "oita", mode="before")
    def pad_radial_float_arrays_mnvolplus(cls, value):
        """
        pad the 1D float arrays to the SPEC expected size
        """
        arr = np.zeros(mnvol + 1, dtype=float)
        src = np.atleast_1d(np.asarray(value, dtype=float))
        arr[:src.size] = src[:arr.size]
        return arr

    @field_validator("helicity", "isurf", mode="before")
    def pad_radial_float_arrays_mnvol(cls, value):
        """
        pad the 1D float arrays to the SPEC expected size
        """
        arr = np.zeros(mnvol, dtype=float)
        src = np.atleast_1d(np.asarray(value, dtype=float))
        arr[:src.size] = src[:arr.size]
        return arr
    
    @field_validator("rac", "zas", "ras", "zac", mode="before")
    def pad_toroidal_arrays(cls, value):
        """
        pad the 1D toroidal arrays to the SPEC expected size
        """
        arr = np.zeros(mntor + 1, dtype=float)
        src = np.atleast_1d(np.asarray(value, dtype=float))
        arr[:src.size] = src[:arr.size]
        return arr

    @field_validator(*arrays_fourier, mode="before")
    def pad_fourier_arrays(cls, value):
        """
        pad the 2D fourier arrays to the SPEC expected size. 
        """
        if hasattr(value, "items"):
            #default, read dict elements and put in array
            return _dict_to_specarray(dict(value))
        elif isinstance(value, np.ndarray):
            # input is an array, pass along for model validator to do fancier things
            # than a simple field validator can do (acces ntor, mpol from other fields)
            value = np.array(value, dtype=float)
            return np.atleast_2d(value)
        else: 
            raise ValueError("Invalid fourier array input.")
    
    @model_validator(mode="after")
    def pad_fourier_arrays_input(self):
        """
        pad the fourier arrays to the spec expected size if they were not suppied as a nested dict
        (for example if read as a namelist or supplied by a different code). 
        The array is expected to be of shape (ntor_input+1, 2*mpol_input+1) (VMEC convention) 
        with negative m values
        stored in the upper half of the second axis.
        """
        provided = self.model_fields_set
        for field in arrays_fourier:
            if field not in provided:
                continue
            current = getattr(self, field)

            if hasattr(current, "items"):
                raise ValueError(f"Field {field} should not be a mapping at this stage.")
            
            try:
                src = np.atleast_2d(np.asarray(current, dtype=float))
            except Exception as exc:
                raise ValueError(f"Cannot convert field {field} to a 2D array.") from exc
            

            if src.shape == (2 * mntor + 1, 2 * mmpol + 1):
                # field already has correct shape, likely from field validator."
                continue
            # if we are here, we are dealing with an array likely from the namelist. 
            arr = np.zeros((2 * mntor + 1, 2 * mmpol + 1))
            mpol_input = src.shape[0] - 1
            ntor_input = (src.shape[1] - 1 ) // 2
            our_ntor = min(ntor_input, self.ntor)
            our_mpol = min(mpol_input, self.mpol)
            for m in range(0, our_mpol + 1):
                for n in range(-our_ntor, our_ntor + 1):
                    if m == 0 and n < 0:
                        continue
                    p_idx_long = mmpol + m
                    t_idx_long = mntor + n
                    n_inarray = ntor_input + n  # input array can be different size
                    arr[t_idx_long, p_idx_long] = float(src[m, n_inarray])
            setattr(self, field, arr)
        return self
    
    @model_validator(mode="before")
    @classmethod
    def check_correct_fields_provided(cls, values):
        """
        ensure that depending on lconstraint, the necessary info was provided
        """
        lconstraint = values.get("lconstraint", -1)
        nvol = values.get("nvol", 1)
        lfreebound = values.get("lfreebound", 0)
        number_of_volumes = nvol + 1 if lfreebound == 1 else nvol  #  also known as mvol

        if lconstraint == 0:
            for field in ["tflux", "pflux", "mu"]:
                assert field in values, f"Field {field} must be provided when lconstraint=0."
                assert len(values[field]) >= number_of_volumes, f"{field} must contain the same amount of elements as there are volumes ({number_of_volumes})  lconstraint=0."
        elif lconstraint == 1:
            for field in ["iota", "oita", "tflux"]:
                assert field in values, f"Field {field} must be provided when lconstraint=1."
            assert len(values["tflux"]) >= number_of_volumes, f"tflux must contain the same amount of elements as there are volumes ( {number_of_volumes})  when lconstraint=1."
            # iota and oita are strange and the first element is actually ignored but must be provided. Should be fixed later
            assert len(values["iota"]) >= number_of_volumes+1, f"iota must contain one more elements as there are volumes ({number_of_volumes}, received {len(values['iota'])})  when lconstraint=1"
            assert len(values["oita"]) >= number_of_volumes+1, f"oita must contain one more elements as there are volumes ({number_of_volumes}, received {len(values['oita'])})  when lconstraint=1"
        elif lconstraint == 2:
            for field in ["tflux", "pflux", "helicity"]:
                assert field in values, f"Field {field} must be provided when lconstraint=2."
                assert len(values[field]) >= number_of_volumes, f"{field} must contain the same amount of elements as there are volumes ({number_of_volumes}, received {len(values[field])})  lconstraint=2."
        elif lconstraint == 3:
            for field in ["tflux", "ivolume", "isurf"]:
                assert field in values, f"Field {field} must be provided when lconstraint=3."
            assert len(values["tflux"]) >= number_of_volumes, f"tflux must contain the same amount of elements as there are volumes ( {number_of_volumes}, received {len(values['tflux'])})  when lconstraint=3."
            assert len(values["ivolume"]) >= number_of_volumes, f"ivolume must contain the same amount of elements as there are volumes ( {number_of_volumes}, received {len(values['ivolume'])})  when lconstraint=3."
            assert len(values["isurf"]) >= number_of_volumes-2, f"isurf must contain one less elements as there are volumes ( {number_of_volumes-1}, received {len(values['isurf'])})  when lconstraint=3."
        return values


class NumericParameters(BaseModel):
    linitialize: int = 0
    lautoinitbn: int = 1
    lzerovac: int = 0
    ndiscrete: int = 2
    nquad: int = -1
    impol: int = -4
    intor: int = -4
    lsparse: int = 0
    lsvdiota: int = 0
    imethod: int = 3
    iorder: int = 2
    iprecon: int = 0
    iotatol: float = -1.0
    mregular: int = -1
    lrzaxis: int = 1
    ntoraxis: int = 3
    lvcgrid: int = 0


class LocalParameters(BaseModel):
    linitgues: int = 1
    lposdef: int = 0
    maxrndgues: float = 1.0


class GlobalParameters(BaseModel):
    lfindzero: int = 0
    escale: float = 0.0
    opsilon: float = 1.0
    pcondense: float = 2.0
    epsilon: float = 0.0
    wpoloidal: float = 1.0
    upsilon: float = 1.0
    forcetol: float = 1.0e-10
    c05xmax: float = 1.0e-06
    c05xtol: float = 1.0e-12
    c05factor: float = 1.0e-02
    lreadgf: bool = True
    mfreeits: int = 0
    bnstol: float = 1.0e-06
    bnsblend: float = 0.666
    gbntol: float = 1.0e-06
    gbnbld: float = 0.666
    vcasingeps: float = 1.0e-12
    vcasingtol: float = 1.0e-08
    vcasingits: int = 8
    vcasingper: int = 1
    vcnt: int = 256
    vcnz: int = 256


class DiagnosticsParameters(BaseModel):
    odetol: float = 1.0e-07
    absreq: float = 1.0e-08
    relreq: float = 1.0e-08
    absacc: float = 1.0e-04
    epsr: float = 1.0e-08
    nppts: int = 0
    ppts: float = 0.0
    nptrj: int | npt.NDArray[np.int32] = Field(default_factory=lambda: np.full(mnvol + 1, -1, dtype=int))
    lhevalues: bool = False
    lhevectors: bool = False
    lhmatrix: bool = False
    dpp: int = -1
    dqq: int = -1
    lerrortype: int = 0
    ngrid: int = -1
    drz: float = 1e-5
    lcheck: int = 0
    ltiming: bool = False
    ltransform: bool = False
    fudge: float = 1.0
    scaling: float = 1.0

    #wrap serializer is not ideal, but poincare will be moved to python
    # so this serializer will disappear and this temporary inefficiency
    # is cleaner than mixing Annotated and wrap styles. 
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _nvol: int = PrivateAttr(default=1)
    _lfreebound: int = PrivateAttr(default=1)
    
    @field_validator("nptrj", mode="before")
    def pad_nptrj(cls, value):
        """
        pad the 1D toroidal arrays to the SPEC expected size
        """
        arr = np.zeros(mnvol + 1, dtype=int)
        src = np.atleast_1d(np.asarray(value, dtype=int))
        arr[:src.size] = src[:arr.size]
        return arr
    
    @field_serializer("nptrj")
    def serialize_nptrj(self, value):
        """
        Serialize the 1D toroidal arrays to the SPEC expected format
        """
        arr = np.atleast_1d(np.asarray(value, dtype=int))
        length = self._nvol + 1 if self._lfreebound == 1 else self._nvol
        if length == 0:
            return arr.tolist()
        return arr[:length].tolist()
    


class ScreenParameters(BaseModel):
    wmanual: bool = False
    wrzaxis: bool = False
    wpackxi: bool = False
    wvolume: bool = False
    wcompute_coordinates: bool = False
    wbasefn: bool = False
    wmemory: bool = False
    wcompute_metric_quantities: bool = False
    wvolume_integrate_chebyshev: bool = False
    wmatrix: bool = False
    wspsint: bool = False
    wsolve_beltrami_system: bool = False
    wconstruct_beltrami_field: bool = False
    wpackab: bool = False
    wcompute_rotational_transform: bool = False
    wcompute_plasma_current: bool = False
    wdf0ab: bool = False
    wlforce: bool = False
    wintghs: bool = False
    wmtrxhs: bool = False
    wlbpol: bool = False
    wbroadcast_profile_data: bool = False
    wdfp100: bool = False
    wdfp200: bool = False
    wdforce: bool = False
    wnewton: bool = False
    wcasing: bool = False
    wbnorml: bool = False
    wcompute_beltrami_error: bool = False
    wconstruct_poincare_plot: bool = False
    wpp00ab: bool = False
    wget_position_vetor_x: bool = False
    whesian: bool = False
    wwrite_vector_potentials: bool = False
    wnumrec: bool = False
    wdcuhre: bool = False
    wminpack: bool = False
    wiqpack: bool = False
    wrksuite: bool = False
    wi1mach: bool = False
    wd1mach: bool = False
    wilut: bool = False
    witers: bool = False
    wsphdf5: bool = False
    winitialize_internal_arrays: bool = False
    wglobal: bool = False
    wxspech: bool = False
    wbuild_vector_potential: bool = False
    wreadin: bool = False
    wwrtend: bool = False
    wmacros: bool = False

class MinimizationParameters(BaseModel, validate_assignment=True): 
    gtol: float = 1e-13
    xtol: float = 1e-22
    ftol: float = 1e-13
    max_nfev: int = 10000 # Large number, by default it's not stopping iterations
    max_niter: int = 10000  # Large number, by default it's not stopping iterations
    method: str = "dogbox"
    rcond_lstsq: float = 1e-10
    save_xin_fixb: bool = True
    freeb_max_iter: int = 1
    freeb_bnorm_tol: float = 1e-5
    nt_dbb: int | None = None
    nz_dbb: int | None = None
    filter_stellsym: bool = True
    objective: str = "freal"
    resume: bool = False
    mpol_rz: int | None = None
    ntor_rz: int | None = None

class InputParameters(BaseModel):
    minimization: MinimizationParameters = Field(default_factory=MinimizationParameters)
    physics: PhysicsParameters = Field(default_factory=PhysicsParameters)
    numeric: NumericParameters = Field(default_factory=NumericParameters)
    local: LocalParameters = Field(default_factory=LocalParameters)
    global_: GlobalParameters = Field(default_factory=GlobalParameters)
    diagnostics: DiagnosticsParameters = Field(default_factory=DiagnosticsParameters)
    #screen: ScreenParameters = Field(default_factory=ScreenParameters)

    @model_validator(mode="after")
    def validate_parameters(self) -> "InputParameters":
        """Placeholder for cross-field validation between parameter groups."""

        # diagnostics needs to know nvol and lfreebound to serialize nptrj correctly, will be removed.
        self.diagnostics._nvol = self.physics.nvol
        self.diagnostics._lfreebound = self.physics.lfreebound

        return self
    

