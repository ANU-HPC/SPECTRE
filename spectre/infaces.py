# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## Work in progress...


def init_interfaces(spec_obj, init_flag=0, outer_inface=None):

    x_init = spec_obj.xin

    if init_flag == 0:
        # read initial interfaces from the input file
        x_new = x_init.copy()

    elif init_flag == 1:
        # implement spectre-like interpolation, linitialize=1 for fixed-b. and linitialize=2 for free-b.
        pass

    elif init_flag == 2:
        # initialize using map2disc

        from spectre import gen_infaces, pack_interfaces

        reven, rodd, zeven, zodd = gen_infaces(spec_obj, x_init, base_inface=outer_inface)

        x_new = pack_interfaces(reven.T, rodd.T, zeven.T, zodd.T, spec_obj)

    else:
        raise RuntimeError(f"Unknown {init_flag=}")

    return x_new
