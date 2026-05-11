# Author: Christopher Berg Smiet on behalf of Proxima Fusion
# @copyright Proxima Fusion 2026
#
# Build context must be the repo root:
#   docker build -f tools/run_container/Dockerfile -t spectre-run .
#
# Requires the CI image to be built and tagged first:
#   docker build -t spectre-ci tools/ci_container/
#
# Override CI_IMAGE to use a registry image instead of the local tag:
#   docker build --build-arg CI_IMAGE=registry.example.com/spectre-ci:latest \
#       -f tools/run_container/Dockerfile -t spectre-run .

ARG CI_IMAGE=spectre-ci

# ── Stage 1: Build SPECTRE on top of the pre-built CI dependency stack ────────
# Re-uses the Spack environment from the CI image so the full dependency stack
# does not need to be rebuilt from source.
FROM ${CI_IMAGE} AS builder

USER root
# gfortran-13 is required at build time to compile the Fortran extensions.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends gfortran-13 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build/spectre
COPY CMakeLists.txt pyproject.toml LICENSE ./
COPY cmake/ cmake/
COPY fortran_src/ fortran_src/
COPY spectre/ spectre/

# Source the Spack env so cmake, mpicc, and gfortran from /opt/spack-view are
# on PATH. --no-build-isolation lets scikit-build-core use Spack's cmake (3.31.x)
# instead of the PyPI cmake wheel (4.x), which has MPI detection issues with
# OpenMPI 5.x. uv is inherited from the CI image via PATH.
# scikit-build-core is removed after the build — it is not needed at runtime.
RUN source /etc/profile.d/z10_spack_environment.sh && \
    export PATH="/home/ubuntu/.local/bin:$PATH" && \
    uv venv --python "$(which python3)" /opt/spectre-env && \
    . /opt/spectre-env/bin/activate && \
    uv pip install --no-cache-dir scikit-build-core "numpy" "f90wrap<0.3" && \
    uv pip install --no-cache-dir --no-build-isolation . && \
    uv pip uninstall scikit-build-core

# ── Stage 2: Final runtime image ──────────────────────────────────────────────
# Start from a clean base and copy only the runtime artifacts from the builder.
# This excludes uv, /opt/spack-environment, and all other build-only content
# that the CI image carries.
FROM spack/ubuntu-noble:latest

USER root

# Runtime libraries: /opt/software contains the Spack-built packages;
# /opt/spack-view is the unified symlink tree that z10_spack_environment.sh
# points PATH and LD_LIBRARY_PATH at. /opt/spack-environment is intentionally
# omitted — it is only needed to run spack commands, not the installed software.
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/spack-view /opt/spack-view
COPY --from=builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh

# Compiled SPECTRE Python environment — no build tools included.
COPY --from=builder /opt/spectre-env /opt/spectre-env

# Run scripts baked in so batch jobs are fully self-contained.
COPY scripts/ /opt/spectre/scripts/
RUN find /opt/spectre/scripts \( -name "*.py" -o -name "*.sh" \) -exec chmod +x {} +

# Profile script: activates spectre-env and adds all scripts subdirectories to
# PATH. Sourced by the entrypoint for every invocation and from ~/.bashrc for
# interactive `docker exec` sessions that bypass the entrypoint.
COPY tools/run_container/z20_spectre.sh /etc/profile.d/z20_spectre.sh

COPY tools/run_container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    echo '. /etc/profile.d/z10_spack_environment.sh' >> /home/ubuntu/.bashrc && \
    echo '. /etc/profile.d/z20_spectre.sh' >> /home/ubuntu/.bashrc && \
    mkdir -p /data && chown ubuntu:ubuntu /data

USER ubuntu
WORKDIR /data

ENV PATH="/opt/spectre-env/bin:$PATH"
ENV SPECTRE_ROOT="/opt/spectre"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
