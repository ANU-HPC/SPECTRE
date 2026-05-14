# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-noble:latest AS builder

# This file is generic and meant to install various ssoftware stack images.
# The stack configurations are stored in the spack_configs directory, and the
# desired stack can be selected by setting the STACK_NAME build argument at
# build time using
# > docker build --build-arg STACK_NAME=<stack_name> [other options] -t <image_name> .
# The stack name is the name of the corresponding spack yaml file without the
# spack_ prefix and .yaml suffix, e.g. gcc-13.3.0 for spack_gcc-13.3.0.yaml.
ARG STACK_NAME=gcc-13.3.0

RUN mkdir /opt/spack-environment
COPY spack_configs/spack_${STACK_NAME}.yaml /opt/spack-environment/spack.yaml
COPY config.yaml /opt/spack/etc/spack/

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

# Strip all the binaries from debug symbols to reduce the size of the image
RUN find -L /opt/software/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Generate the commands to activate the Spack environment when a new shell is opened
RUN cd /opt/spack-environment && \
spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# Bare OS image to run the installed executables
FROM spack/ubuntu-noble:latest

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/spack-view /opt/spack-view
COPY --from=builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh


# Wrapper sources the Spack activation script before exec'ing the command,
# so the env is active for non-login invocations too (docker run img cmd).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ubuntu
WORKDIR /home/ubuntu

# Install UV tool
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ubuntu/.local/bin/:$PATH"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
