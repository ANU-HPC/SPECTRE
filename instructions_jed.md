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
    `python -m venv --system-site-packages ~/virtualenvs/venv-spectre`

3. Init virtual environment
    `source ~/virtualenvs/venv-spectre/bin/activate`

4. Clone gitlab in a good directory
    `git clone git@gitlab.com:spectre-eq/spectre.git`

5. Enter directory and pip install
    `pip install -e .` 

6. Create initialization file `~/src_spectre.sh` with the following:
    ```
    module purge
    module load gcc
    module load openmpi
    module load openblas
    module load hdf5/1.14.3-mpi
    module load fftw/3.3.10-mpi-openmp

    module load python
    module load py-numpy py-pip py-mpi4py

    source ~/virtualenvs/venv-spectre/bin/activate

    export PATH=$PATH:~/codes/spectre/scripts/
    ```

7. Run the example QA force minimization in `examples/qa_vac_vol7`

