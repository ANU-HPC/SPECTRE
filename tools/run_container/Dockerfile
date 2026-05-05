# Author: Christopher Berg Smiet on behalf of Proxima Fusion
# @copyright Proxima Fusion 2026
#
# Build context must be the repo root:
#   docker build -f tools/run_container/Dockerfile -t spectre-run .

# ── Stage 1: Build the Spack dependency stack ─────────────────────────────────
FROM spack/ubuntu-noble:latest AS spack-builder
# Inherit the base image's SHELL (["/bin/bash", "--login", "-c"]) so that
# /etc/profile.d/ is sourced and spack is on PATH for every RUN command.

# Ensure gfortran is available and register the system GCC with spack.
# This avoids compiling GCC from source (saves ~40 min per build).
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends gfortran-13 && \
    rm -rf /var/lib/apt/lists/* && \
    spack compiler find

# Add the public Spack binary mirror to download pre-built packages where available.
RUN spack mirror add spack-public https://binaries.spack.io/releases/v0.23 && \
    spack buildcache keys --install --trust

RUN mkdir /opt/spack-environment
COPY tools/run_container/spack.yaml /opt/spack-environment
COPY tools/run_container/config.yaml /opt/spack/etc/spack/

# Install the dependency stack, remove stale packages, strip binaries.
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

RUN find -L /opt/software/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Generate the environment activation script (sets PATH, LD_LIBRARY_PATH, etc.)
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# ── Stage 2: Build and install SPECTRE ────────────────────────────────────────
FROM spack-builder AS spectre-builder

SHELL ["/bin/bash", "-c"]

# Copy only the files needed to build the package.
WORKDIR /build/spectre
COPY CMakeLists.txt pyproject.toml LICENSE ./
COPY cmake/ cmake/
COPY fortran_src/ fortran_src/
COPY spectre/ spectre/

# Install uv then source the Spack env and build in one shell so PATH stays consistent.
# Source the Spack env: the view at /opt/spack-view makes cmake, mpicc, gfortran, python3
# etc. available on PATH and LD_LIBRARY_PATH.  Pre-install the Python build deps then
# use --no-build-isolation so scikit-build-core picks up spack's cmake 3.31.x from PATH
# instead of the PyPI cmake wheel (4.x), which has MPI detection issues with OpenMPI 5.x.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source /etc/profile.d/z10_spack_environment.sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv venv --python "$(which python3)" /opt/spectre-env && \
    source /opt/spectre-env/bin/activate && \
    uv pip install --no-cache-dir scikit-build-core "numpy" "f90wrap<0.3" && \
    uv pip install --no-cache-dir --no-build-isolation .

# ── Stage 3: Final runtime image ──────────────────────────────────────────────
FROM spack/ubuntu-noble:latest

# ubuntu:noble ships with an 'ubuntu' user at UID 1000; use it directly.

# Runtime libraries from Spack (openmpi, hdf5, openblas, fftw, Python, and their
# transitive deps). The activation script sets PATH and LD_LIBRARY_PATH so the
# compiled Fortran extensions can find their shared libraries without a login shell.
COPY --from=spack-builder /opt/spack-environment /opt/spack-environment
COPY --from=spack-builder /opt/software /opt/software
# The view contains symlinks into /opt/software; the activation script exports
# PATH and LD_LIBRARY_PATH pointing here, so mpiexec/python3/libs are findable
# at runtime without a module system.
COPY --from=spack-builder /opt/spack-view /opt/spack-view
COPY --from=spack-builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh

# Installed SPECTRE Python environment
COPY --from=spectre-builder /opt/spectre-env /opt/spectre-env

# SPECTRE run scripts baked into the image so batch jobs are self-contained
COPY scripts/ /opt/spectre/scripts/

# Entrypoint: sources Spack runtime env then execs the requested command.
# This is the correct pattern for Cloud Run / Cloud Batch — avoids hardcoding
# content-hashed Spack paths into ENV while adding <1 ms overhead.
COPY tools/run_container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /data && chown ubuntu:ubuntu /data

USER ubuntu
WORKDIR /data

ENV PATH="/opt/spectre-env/bin:$PATH"
ENV SPECTRE_ROOT="/opt/spectre"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
