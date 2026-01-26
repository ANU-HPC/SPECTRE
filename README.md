# SPECTRE 

SPECTRE (Stepped Pressure Equilibrium Code The Revised Edition) is a tool for calculating general 3D MHD equilibrium solutions with generalized magnetic topology. This is done in the framework of the Multi-Region relaxed MHD (MRxMHD) model which solves plasma subdivided into stepped-pressure subdomains, separated by the ideal interfaces. The code is built upon a previous code SPEC, but includes significant improvements.

- Original code developers: Erol Balkovic, Joaquim Loizu, Emmanuel Lanti, Chris Smiet
- A list of related publications ...
- Please direct questions/suggestions to ...

## Installation

The code can be simply installed using `pip install -e .` (for editable install) in the root directory.


## Citation
If you use SPECTRE in your research, please cite it appropriately using the "Cite this repository" button on the right ...

## Documentation
Documenation for the code is availabe in .url.

## Debugging Fortran
Add following line to `[tool.scikit-build.cmake.define]` in `pyproject.toml` for debugging with gfortran:
`CMAKE_Fortran_FLAGS_DEBUG="-g -O0 -fcheck=bounds -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"`
