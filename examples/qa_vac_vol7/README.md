## Running SPECTRE

The included files are:
- `init.toml` which contains the inputs needed to run spectre for a 7-volume vacuum QA configuration, and is generated from `init.sp` using `convert_sp_to_toml.py`
- `analysis_qa_vac_vol7.ipynb` which is a jupyter notebook that can be used to analyze the results


To calculate the initial field (before optimizing interfaces for Force balance) one can run `calc_spectre_field.py init.toml`. This produces: 
- `init_end.toml` which is an end file that can be used for restart
- `init.h5` which is a hdf5 file used for analysis/postprocessing


To actually find the interfaces which correspond to an equilibrium and satisfy the zero-pressure-jump condition, one runs spectre using the input file `init.toml`. This can be either done:
1. Locally: running `calc_spectre_eq.py init.toml` (potentially with mpi) 
2. Using slurm: submit a slurm job that runs spectre. An example slurm submission script is given and can be run as `sbatch slurm_job.sh init.toml`. Note that details on `#SLURM` directives are cluster dependent. 


During force minimization output is piped into a text file, and one can monitor progress using scripts `printout.sh`, `plot_x_single.py`, and `plotout.py`.


Once optimization is done, there will be a generated `init_opt.toml` input file which contains the optimized interfaces. In this file, one can choose to do poincare plotting (setting the fields in `diagnostics`). Then, the Beltrami field in this final state is calculated by running `calc_spectre_field.py init_opt.toml` which generates another output file `init_opt.h5` (which is also directly generated after optimization). 


From this point analysis can be done (using SPECTREout class), as demonstrated in the included jupyter notebook.

Note that spectre also allows for in-directory restarts. By changing parameter `max_niter` and rerunning in the same directory, spectre will automatically figure out there are previous iterations and will continue from those.