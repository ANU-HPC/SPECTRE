## Setting up SPECTRE on the Pitagora cluster from Cineca

1. Load modules
    ```
    module purge

    module load gcc
    module load openmpi/4.1.6--gcc--12.3.0
    module load openblas/0.3.26--gcc--12.3.0
    module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.3.0
    module load fftw/3.3.10--openmpi--4.1.6--gcc--12.3.0
    module load python python-venv/1.0-4vooo3n py-mpi4py/3.1.5--openmpi--4.1.6--gcc--12.3.0  py-numpy/1.26.4--gcc--12.3.0-punolzs py-pip/23.0--gcc--12.3.0-2ulj5zc py-scipy/1.13.0--gcc--12.3.0-3fgx7rh
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
    module load openmpi/4.1.6--gcc--12.3.0
    module load openblas/0.3.26--gcc--12.3.0
    module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.3.0
    module load fftw/3.3.10--openmpi--4.1.6--gcc--12.3.0
    module load python python-venv/1.0-4vooo3n py-mpi4py/3.1.5--openmpi--4.1.6--gcc--12.3.0  py-numpy/1.26.4--gcc--12.3.0-punolzs py-pip/23.0--gcc--12.3.0-2ulj5zc py-scipy/1.13.0--gcc--12.3.0-3fgx7rh

    source ~/virtualenvs/venv-spectre/bin/activate

    export PATH=$PATH:~/codes/spectre/scripts/
    ```

7. Run the example QA force minimization in `examples/qa_vac_vol7`

#### Technically, the py-* modules loaded above are 'hidden' and should not be used. They can be alternatively 'pip installed', but usually these precompiled modules are more optimized (not checked).