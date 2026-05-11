# SPECTRE run container

This folder contains the configuration files to build a Docker image with SPECTRE pre-installed and ready to use.

The image is built in two stages to minimise its final size:

1. **`builder`** — compiles and installs SPECTRE (Fortran extensions + Python package) on top of `elanti/spec-ci-container`, placing the result in `/opt/spectre-env`.
2. **Final image** — a fresh copy of `elanti/spec-ci-container` with only the compiled `/opt/spectre-env` and the run scripts added; no build tools or source code.

## How to build

The build context must be the **repository root** so that the SPECTRE source is available:

```bash
docker build -f tools/run_container/Dockerfile -t spectre-run .
```

## How to run

Interactive shell (SPECTRE and all runtime libraries are on the path):

```bash
docker run -it --rm -v $(pwd):/data spectre-run
```

Run a script directly:

```bash
docker run --rm -v $(pwd):/data spectre-run python /data/my_script.py
```

## Shell environment

Every session — interactive or batch — gets:

- The Spack runtime libraries on `PATH` and `LD_LIBRARY_PATH` (via `z10_spack_environment.sh`).
- The SPECTRE Python environment activated (via `z20_spectre.sh`).
- All scripts subdirectories (`scripts/run`, `scripts/setup`, etc.) on `PATH` so run scripts are directly callable.
- `SPECTRE_ROOT=/opt/spectre`.

This is set up by the entrypoint for normal `docker run` invocations, and by `/home/ubuntu/.bashrc` for `docker exec -it <container> bash` sessions.
