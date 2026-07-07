# Force-Jacobian OpenMP scaling analysis

VTune investigation of why the parallelized force-Jacobian build
(`solve_field_deriv`, branch `jurasic-parallelize-pure-func-2`) plateaus at ~2.4×
speedup on 6 threads instead of scaling further.

## TL;DR

- The DOF loop is now correctly parallelized (loop 1 = RHS build, loop 3 =
  unpack+derivatives) and is **not** limited by thread synchronization: VTune reports
  **Spin Time ≈ 0.01 s, Overhead ≈ 0 s**. Threads are always busy; the batched
  `DGETRS` serial section is only ~4 % of runtime.
- The scaling ceiling is **memory-bandwidth contention**. Total CPU-time grows ~4×
  going 1→6 threads for the *same* work — the compute kernels keep a low, flat CPI
  (~0.3) but the memory subsystem saturates.
- The single largest bandwidth consumer is **`memset` — 16 % of all CPU time at
  CPI 57–65** (i.e. cores stalled ~60 cycles/instruction on memory). It comes from
  `compute_coordinates` zeroing its full metric output arrays
  (`Rij/Zij/sg/guvij`, ~4 MB/call at `Ntz = 4096`) on **every** call, at every
  quadrature point of every DOF.
- This zeroing is pre-existing baseline behavior (not introduced by the
  purification); the parallelization merely exposed it as the bottleneck.

## Hardware / environment

- CPU: Intel Core Ultra 7 155H (Meteor Lake, hybrid: 16 physical cores / 22 logical;
  ~6 performance cores — consistent with the speedup plateau at 6 threads).
- VTune 2025.1, driverless `perf` collection.
- Benchmark case: `examples/analysis/finding_fixed_points/test_qa_nvol7.toml`
  (`Mvol = 7`, `mn = 145`, `Nt = Nz = 64` → `Ntz = 4096`).

## Speedup (clean machine, best-of-3, threads pinned)

`OMP_PROC_BIND=close OMP_PLACES=cores`, both loops parallel:

| Threads | Time  | Speedup |
|--------:|------:|--------:|
| 1       | 30.6s | 1.00×   |
| 2       | 19.9s | 1.54×   |
| 4       | 14.6s | 2.09×   |
| 6       | 12.6s | 2.43×   |
| 8       | 13.4s | 2.28×   |
| 12      | 13.6s | 2.25×   |

Amdahl fit over N≤6: parallel fraction **p ≈ 0.70**, ceiling ≈ 3.4×. The regression
at 8/12 threads is the hybrid CPU (threads spill onto slower E-cores) plus the
bandwidth wall, not pure Amdahl (which is monotonic).

## VTune hotspots — the smoking gun

HW-event hotspots (`hotspots -knob sampling-mode=hw`), OMP=1 vs OMP=6, per-function
self CPU time and CPI:

| Function                     | CPU@1  | CPU@6  | CPU ratio | CPI@1 | CPI@6 |
|------------------------------|-------:|-------:|----------:|------:|------:|
| `compute_coordinates`        | 11.2 s | 49.3 s | 4.4×      | 0.32  | 0.34  |
| **`__memset_avx2…`**         | 10.9 s | 46.4 s | 4.2×      | **57**| **65**|
| `dgemm_kernel` (OpenBLAS)    |  7.6 s | 29.7 s | 3.9×      | 0.30  | 0.36  |
| `volume_integrate_chebyshev` |  7.3 s | 20.7 s | 2.8×      | 0.34  | 0.39  |
| `intghs`                     |  4.3 s | 18.9 s | 4.4×      | 0.35  | 0.36  |
| `invfft`                     |  3.4 s | 13.9 s | 4.1×      | 0.32  | 0.33  |

OMP=6 run summary: Elapsed 81.9 s, CPU 287.2 s, CPI 0.46, Spin 0.01 s, Overhead 0 s.
`memset` + `memmove` together are ~19 % of all CPU time and run at CPI 60–108 —
pure memory-stall.

Interpretation:
- Compute kernels (`compute_coordinates` arithmetic, `intghs`, `invfft`, `dgemm`)
  have **CPI ~0.3 that barely rises** 1→6 threads — they scale well and are not the
  problem.
- Every function's *total CPU-seconds* grows ~4× (identical work, 4× the cycles):
  the concurrent `memset`s saturate the memory controller, so all memory-touching
  work slows down per byte under contention.
- The `memset` is the array zeroing at the top of `compute_coordinates`
  (`geometry_mod.F90:48-57`): `Rij/Zij(1:Ntz,0:3,0:3)` and
  `guvij(1:Ntz,1:3,1:3,·)` are zeroed in full on every call, then largely
  overwritten. That is the redundant traffic to remove.

## Root cause (code)

`fortran_src/geometry_mod.F90`, `compute_coordinates`:

```fortran
Rij(1:Ntz, 0:3, 0:3) = 0.0_rkind          ! 16 slots zeroed every call
Zij(1:Ntz, 0:3, 0:3) = 0.0_rkind
if (Lcurvature > 0) then
  sg(1:Ntz, 0) = 0.0_rkind
  inv_jac(1:Ntz) = 0.0_rkind
  guvij(1:Ntz, 1:3, 1:3, 0) = 0.0_rkind
end if
if (Lcurvature > 1) then
  sg(1:Ntz, 1:3) = 0.0_rkind
  guvij(1:Ntz, 1:3, 1:3, 1:3) = 0.0_rkind
end if
```

Called once per Gauss quadrature point per DOF, so at very high frequency. Only the
slots that are *accumulated into* (rather than directly assigned) actually need
pre-zeroing; the rest is wasted bandwidth.

## Fixes (highest leverage first)

1. **Trim the zeroing in `compute_coordinates`** to only the slots that are read
   before being written (accumulator slots), converting the rest to direct
   assignment. Attacks the 16 % `memset` directly; helps both serial and parallel.
2. **Reuse per-thread metric scratch** across quadrature points where the overwrite
   is total, instead of reallocating/zeroing each call.
3. Longer term: parallelize the outer interface loop so the serial `serial_rest`
   (LHS build, LU factorization, workspace init) overlaps across interfaces.

None require changing the parallel structure.

## Reproduction

### 1. Prerequisites (one-time, root)

Driverless VTune needs relaxed kernel knobs:

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0     # allow attach/sampling
sudo sysctl -w kernel.perf_event_paranoid=1   # HW event-based sampling
sudo sysctl -w kernel.kptr_restrict=0         # resolve kernel symbols
```

(SW hotspots need only `ptrace_scope=0`. Memory-*bandwidth* uncore counters
additionally require the VTune SEP sampling driver — not needed for the analysis
above, which used HW hotspots + CPI.)

### 2. Build the wheel with debug info (keep -O3)

```bash
SKBUILD_CMAKE_ARGS="-DCMAKE_Fortran_FLAGS=-O3 -g -fno-omit-frame-pointer;-DCMAKE_C_FLAGS=-O3 -g -fno-omit-frame-pointer -DNDEBUG" \
  uv pip install --python .venv/bin/python --no-build-isolation --force-reinstall --no-deps -e .

# scikit-build strips the installed .so on copy; overwrite it with the unstripped
# build/ copy so VTune can resolve source lines:
cp build/spectre/_spectre4py_wrap.*.so \
   .venv/lib/python3.*/site-packages/_spectre4py_wrap.*.so
```

### 3. Profiling driver

`prof_jac.py` — warm up once (FFTW plan creation), then time N `force_real_jac`
calls so the profile is dominated by the Jacobian build:

```python
import os, sys, time, numpy as np
from spectre import SPECTRE, get_xinit_specwrap, force_real, force_real_jac
test = SPECTRE("examples/analysis/finding_fixed_points/test_qa_nvol7.toml", verbose=False)
x = get_xinit_specwrap(test); force_real(x, test)
force_real_jac(x, test)                       # warmup
t0 = time.perf_counter()
for _ in range(int(sys.argv[1])):
    J = force_real_jac(x, test)
print(f"OMP={os.environ.get('OMP_NUM_THREADS')} avg={(time.perf_counter()-t0)/int(sys.argv[1]):.3f}s")
```

### 4. Collect

```bash
source /opt/intel/oneapi/vtune/latest/env/vars.sh

# OMP=6 (2 reps is enough; stack collection off keeps it fast)
OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  vtune -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=false \
  -r /tmp/vt_hs6 -- .venv/bin/python prof_jac.py 2

# OMP=1 baseline for the CPI/CPU-ratio comparison
OMP_NUM_THREADS=1 \
  vtune -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=false \
  -r /tmp/vt_hs1 -- .venv/bin/python prof_jac.py 1
```

### 5. Report

```bash
# per-function CPU time + CPI (column 2 = CPU Time, column 9 = CPI Rate)
vtune -report hotspots -r /tmp/vt_hs6 -format csv -csv-delimiter=comma | \
  awk -F, '{print $1"|CPU="$2"|CPI="$9}' | head -20

# run summary (Spin/Overhead/CPI/core utilization)
vtune -report summary -r /tmp/vt_hs6 | \
  grep -iE "Elapsed|CPU Time:|Spin|Overhead|CPI|Core Utilization"
```

Look for: near-zero Spin/Overhead (rules out sync issues), `memset` high in the list
with CPI ≫ 1 (memory-stall), and total per-function CPU-time growing ~Nthreads×
(work not scaling → bandwidth bound).

### Cleanup

```bash
git worktree remove <baseline-worktree> --force   # if used
# restore the normal optimized wheel:
uv pip install --python .venv/bin/python --no-build-isolation --force-reinstall --no-deps -e .
```
