# SPECTRE 

SPECTRE (Stepped Pressure Equilibrium Code The Revised Edition) is a tool for calculating general 3D MHD equilibrium solutions with generalized magnetic topology. This is done in the framework of the Multi-Region relaxed MHD (MRxMHD) model which solves plasma subdivided into stepped-pressure subdomains, separated by the ideal interfaces. The code is built upon a previous code SPEC, but includes significant improvements.

- Original code developers: Erol Balkovic, Joaquim Loizu, Emmanuel Lanti, Chris Smiet
- A list of related publications ...
- Please direct questions/suggestions to ...

## Installation
The code can be simply installed using `pip install -e .` (for editable install) in the root directory. For examples of installation on some systems, check out `instructions_*.md`

## Testing
To verify that the code compiled and works correctly, you can run the tests in the `tests/` directory by simply running `pytest` in the root directory.

## Example of finding an equilibrium
Look at the `examples/qa_vac_vol7` directory for an example of a force minimization calculation in a 7-volume QA stellarator vacuum equilibrium.

## Documentation
Documenation for the code is availabe...

## Debugging Fortran
Add following line to `[tool.scikit-build.cmake.define]` in `pyproject.toml` for debugging with gfortran:
`CMAKE_Fortran_FLAGS_DEBUG="-g -O0 -fcheck=bounds -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"`

## Citation
If you use SPECTRE in your research, please cite it appropriately using the "Cite this repository" button on the right ...