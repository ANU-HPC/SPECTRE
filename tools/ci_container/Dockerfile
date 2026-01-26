# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-bionic:latest AS builder

# Install newer compiler
RUN spack install gcc@13.2.0 && spack compiler add --scope site $(spack location -i gcc@13.2.0)

# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir /opt/spack-environment
COPY spack.yaml /opt/spack-environment
COPY config.yaml /opt/spack/etc/spack/

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

# Strip all the binaries
RUN find -L /opt/software/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Solve problem on libgcc.a
#RUN ranlib $(spack location -i gcc@13.2.0)/lib/gcc/x86_64-pc-linux-gnu/13.2.0/libgcc.a

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# Regenerate module file
COPY modules.yaml /opt/spack/etc/spack/
RUN cd /opt/spack-environment && \
    spack env activate . && \
    spack module lmod refresh -y

# Copy init files from lmod
RUN ln -s $(spack location -i lmod)/lmod/lmod/init/profile /etc/profile.d/z00_lmod.sh
COPY z01_lmod.sh /etc/profile.d

# Bare OS image to run the installed executables
FROM spack/ubuntu-bionic:latest

RUN adduser --uid 1000 --shell /bin/bash app

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /etc/profile.d /etc/profile.d

RUN mkdir -p /code && chown app:app /code

USER app
WORKDIR /home/app

# Install UV tool
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/app/.local/bin/:$PATH"

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
