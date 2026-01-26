# SPECTRE CI container

This folder contains the configuration files to build the CI Docker image. It contains the software stack used to compile SPECTRE.

## How to build

Use the Docker `build` command:

```bash
docker build -t spectre-ci-container .
```

where `spectre-ci-container` is the name of the generated image.

## How to run

```bash
docker run -it --rm -v $(pwd):/workspace spectre-ci-container
```

## How does it work

This configuration builds a Docker image that contains a software stack built using Spack. While this initial configuration is
rather simple, it is done with extensibility in mind. The main idea is to be able to add various compilers and libraries and Spack
will take care of compiling everything in a consistent way. The result is a Docker image that when run gives you access to all the
required dependencies to build SPECTRE/SPEC4py. They are presented in the form of standard Lmod modules that you can load as you would
typically do on an HPC cluster.

The different packages to install are defined in the `spack.yaml` file. It defines three variables:

- `core-packages`: This is the list of packages that need to be compiled once by any compiler. One could typically find CMake and
  other non-compiler dependent tools. By default they are built using GCC `v13.2.0`.
- `packages`: This is the list of packages that are compiler dependent.
- `compilers`: This is the list of compilers to use to build the stack.