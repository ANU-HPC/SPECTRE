from json import load, dump
from pickle import load as pkl_load, dump as pkl_dump
from numpy import array, ndarray

def _nested_to_array(obj):
    """
    Recursively convert nested lists into numpy arrays as deep as possible
    until inhomogeneity is encountered.
    """
    if not isinstance(obj, (list, tuple)):
        return obj

    try:
        arr = array(obj)
        if arr.dtype != object:
            return arr
    except Exception:
        pass

    return [_nested_to_array(x) for x in obj]


class input_dict(dict):
    """input dictionary class
    -> with dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __new__(cls, *args, **kwargs):
        obj = dict().__new__(cls)
        return obj

    def __init__(self, fname=None):
        if fname is not None:
            if isinstance(fname, str):

                if fname[-4:] == ".pkl":
                    with open(fname, "rb") as f:
                        d = pkl_load(f)
                elif fname[-5:] == ".json":
                    with open(fname, "r") as f:
                        d = load(f)
                else:
                    raise RuntimeError("Input file has to be either .pkl or .json")
            
            elif isinstance(fname, dict):
                d = fname
            self.update(d)

            for k in self.keys():
                if isinstance(self[k], list):
                    self[k] = _nested_to_array(self[k])

    def set_if_none(self, key, val):
        if key not in self.keys():
            self[key] = val

    def has_key(self, key):
        if key in self.keys():
            return True
        else:
            return False

    def copy(self, memo=None):
        from copy import deepcopy

        return input_dict(deepcopy(dict(self), memo=memo))

    def get_keys(self):
        return list(self.keys())

    def __dir__(self):
        return (
            dict().__dir__()
            + self.get_keys()
            + ["get_keys", "save_json", "save_pickle", "copy", "has_key", "set_if_none"]
        )

    def save_json(self, fname, force=False):
        def default(obj):
            if isinstance(obj, ndarray):
                return obj.tolist()
            else:
                return None
            # raise TypeError(f'Not serializable {obj}')

        with open(fname, "w" if force else "x") as f:
            dump(self, f, default=default, indent=4, sort_keys=True)

    def save_pickle(self, fname, force=False):
        with open(fname, "wb" if force else "xb") as f:
            pkl_dump(dict(self), f)
