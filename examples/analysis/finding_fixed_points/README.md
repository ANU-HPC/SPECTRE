# Fixed-point tracking example

Tracks the magnetic axis and the **3/5 island chain** (volume 3) of a fixed-boundary **NCSX**
6-volume equilibrium (`Nfp = 3`) across a force-minimisation run, and plots how the axis and the
island O-point move and how the island's Greene residue evolves.

Only the input `init.toml` is checked in — you generate the data by running the calculation.

## Generate the equilibrium input

`init.toml` was produced from the public NCSX VMEC reference
`tests/test_files/wout_li383_low_res_reference.nc` with `simsvmec2spectre`:

```python
from simsopt.mhd import Vmec
from spectre.converters.simsvmec2spectre import simsvmec2spectre

simsvmec2spectre(
    Vmec("wout_li383_low_res_reference.nc"),
    nvol=6, mpol=7, ntor=6, interface_guess="wout", angle_fix=True,
    filename="init.toml",
)
```

The `[minimization]` block then enables in-loop fixed-point tracking of the 3/5 chain in volume 3
(the 3/5 surface starts near the volume-3/4 interface and moves into volume 3 as the run relaxes):

```toml
track_axis = true
track_islands = true
island_volume = 3
island_m = 5
island_n = 3
```

With no `island_guess_rz` the chain is self-discovered by a fixed-point scan.

## Regenerate the data

Run the optimisation (one MPI rank per volume; `nvol = 6`):

```sh
mpirun -n 6 calc_spectre_eq.py init.toml
```

This writes, in this directory:

- `init_opt.h5` — the optimised equilibrium. Its `/iterations/fp` group holds the per-iterate
  axis and island metrics (the **in-loop** path).
- `xs_fixb/x_<seg>_<iter>.npy` — a saved interface checkpoint per accepted iterate.

The notebook reads a small `fp_metrics.npz`. Produce it from the saved checkpoints with the
**offline** driver (no re-optimisation — it replays the checkpoints):

```sh
python ../../../scripts/minimization_monitoring/track_fp.py \
    init.toml -v 3 -m 5 -n 3 \
    --data-files xs_fixb/*.npy \
    --output fp_metrics.npz
```

The chain is named by `-m`/`-n` (`iota = n/m = 3/5`) and its volume by `-v`; with no `--guess`
the tracker self-seeds by discovery.

Alternatively, read the in-loop `/iterations/fp` group of `init_opt.h5` directly — it carries the
same arrays (`axis_rz`, `island_rz`, `greene_residue`, `iota`, …).

## Plot

Open `example.ipynb`. It loads `fp_metrics.npz` and plots the axis trajectory, the island O-point
trajectory, and the Greene residue over the optimisation iterates. The full list of arrays and
metadata in the `.npz` (identical to the h5 `/iterations/fp` schema) is documented in
`docs/source/user_guide/analysis/fixed_point_tracking.rst`.

## Interactive use

Both finders are also methods on `SPECTREout`, so a single equilibrium (e.g. the final
`init_opt.h5`) can be inspected without any tracking setup:

```python
from spectre import SPECTREout

out = SPECTREout("init_opt.h5")
out.find_axis(guess=[1.4, 0.0])
members = out.find_fixed_point(3, 5, return_all=True)  # discover the 3/5 chain
fp = out.find_fixed_point(3, 5, guess_rz=(members[0].R, members[0].Z))
```
