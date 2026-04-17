#####
Fixed-boundary QA vacuum
#####

This is an example calculation for a 7-volume Fixed-boundary QA vacuum equilibrium. The relevant files are in ``examples/optimization/qa_vac_vol7`` with the input file ``init.toml`` which includes the profiles and initial interface geometry which is obtained by interpolating the plasma boundary inwards. 

The flag ``Lconstraint=3`` means that the invariant profiles are: 

1. Pressure (in each subvolume)
2. Toroidal flux ``tflux`` (in each subvolume)
3. Integrated toroidal current ``ivolume`` (in each subvolume)
4. Integrated toroidal surface current ``isurf`` (on each interface)

This is a vacuum case so the currents and pressures are 0.

The calculation is run using the command ``calc_spectre_field.py init.toml``. This can be done:
1. Locally: for instance using the script ``local_job.sh``
2. Using job scheduling: with slurm on a cluster as is done in the script ``slurm_job_jed.sh``. Note that details on #SLURM directives are cluster dependent.

During optimization, one can monitor the progress using scripts in ``scripts/minimization_monitoring`` which allows to track iterations and look at intermediate states.

After the equilibrium calculation, the following files are generated: 

- ``init.h5``: output hdf5 file with the final interfaces, field, and various diagnostics
- ``init_opt.toml``: a restart toml file that is equivalent to ``init.toml``, except it contains the optimized interface geometry which is now in force balance
- ``out.txt``: a text file with the output log, including the iteration information along with the initial and final error in the field and force

To analyse the result, one uses ``init_opt.h5`` which contains the equilibrium field and allows for analysis and plotting using the class ``SPECTREout`` or scripts in ``scripts/postprocess``. Alternatively, one can use ``init_opt.toml`` for further optimization, paramater changes, or additional Poincare tracing.

.. toctree::
   :maxdepth: 1

   analysis_qa_vac_vol7