## Setting up SPECTRE on an SPC computer using conda

1. `module purge`

2. `conda create -n spectre_env`

3. `conda activate spectre_env`

4. `conda install python==3.13.2 pip gfortran_linux-64=11 libgfortran=14 openblas mpi4py openmpi  fftw hdf5`

5. Exit and reopen terminal

6. Clone gitlab in a good directory
    `git clone git@gitlab.epfl.ch:spec/spec.git`

7. Enter directory and pip install
    `pip install -e .` 

8. Create initialization file `~/src_glspec.sh` with the following:
    ```
    module purge

    conda activate spec_gitlab

    export PATH=$PATH:~/codes/spectre/scripts/
    ```

9. Run the example QA in `spec/examples/qa_vac_vol7` by running the script `run_jed.py`

