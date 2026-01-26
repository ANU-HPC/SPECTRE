## Setting up SPECTRE on JED

1. Load modules
    ```
    module purge
    module load gcc
    module load openmpi
    module load openblas
    module load hdf5/1.14.3-mpi
    module load fftw/3.3.10-mpi-openmp

    module load python
    module load py-numpy py-pip py-mpi4py
    ```

2. Create virutal environment
    ` python -m venv --system-site-packages ~/virtualenvs/venv-spec`

3. Init virtual environment
    `source ~/virtualenvs/venv-spec/bin/activate`

4. Clone gitlab in a good directory
    `git clone git@gitlab.epfl.ch:spec/spec.git`

5. Enter directory and pip install
    `pip install -e .` 

6. Create initialization file `~/src_glspec.sh` with the following:
    ```
    module purge
    module load gcc
    module load openmpi
    module load openblas
    module load hdf5/1.14.3-mpi
    module load fftw/3.3.10-mpi-openmp

    module load python
    module load py-numpy py-pip py-mpi4py

    source ~/virtualenvs/venv-spec/bin/activate

    export PATH=$PATH:~/codes/spectre/scripts/
    ```

7. Run the example QA in `spec/examples/qa_vac_vol7` by running the script `run_jed.py`

