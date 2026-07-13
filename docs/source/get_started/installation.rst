############
Installation
############

Code architecture
=================

SPECTRE is a hybrid **Fortran/Python** code. The Fortran sources are found in
``fortran_src``, while the Python core lies in ``spectre``. Python bindings to
the Fortran routines are generated automatically at build time by
`f90wrap <https://github.com/jameskermode/f90wrap>`_, so the Fortran
subroutines can be called directly from Python.

The build system uses CMake together with scikit-build-core, which means a
single ``pip install`` command compiles the Fortran code, generates the Python
bindings, and installs the Python package.

Prerequisites
=============

System tools
------------

The following tools must be available on your system before installing:

* **Git**
* **CMake** ≥ 3.17
* **Fortran compiler** — GNU Fortran (``gfortran``) is recommended; Intel
  LLVM (``ifx``) is also supported.
* **Python** ≥ 3.10

C/Fortran libraries
-------------------

These libraries must be installed system-wide or made available through a
module system or conda before running ``pip install``:

+--------------------+-----------------------------------------------+
| Library            | Notes                                         |
+====================+===============================================+
| **MPI**            |                                               |
+--------------------+-----------------------------------------------+
| **LAPACK/OpenBLAS**| Build with threading enabled (Spack:          |
|                    | ``openblas threads=openmp``). The dense       |
|                    | Beltrami solves are BLAS-bound; a serial      |
|                    | OpenBLAS silently ignores                     |
|                    | ``OPENBLAS_NUM_THREADS``                      |
+--------------------+-----------------------------------------------+
| **HDF5**           | Must be compiled with MPI and Fortran support |
+--------------------+-----------------------------------------------+
| **FFTW3**          | Needed for the default FFT backend only (not  |
|                    | for ``SPECTRE_COMMERCIAL_USE`` / ``pocketfft``|
|                    | builds, see below)                            |
+--------------------+-----------------------------------------------+

Python dependencies
-------------------

The following Python packages are installed automatically when you run
``pip install``:

* ``numpy``, ``scipy``, ``numba``, ``pydantic``
* ``f90wrap < 0.3`` — generates the Fortran–Python bindings
* ``mpi4py``, ``h5py``
* ``matplotlib``, ``CyRK``
* ``f90nml``, ``tomlkit``, ``threadpoolctl``

Generic installation
====================

Once the system prerequisites above are satisfied, clone the repository and
install with pip:

.. code-block:: bash

   git clone git@gitlab.com:spectre-eq/spectre.git
   cd spectre
   pip install -e .

The ``-e`` flag installs in *editable* mode, which is recommended for
development: source-file changes in ``spectre/`` take effect immediately
without reinstalling. Note that changes to the Fortran sources in
``fortran_src/`` will require re-running ``pip install`` to recompile the
Fortran code.

During ``pip install``, CMake compiles all Fortran modules and f90wrap
generates the Python interface automatically.

.. note::

   If the build fails with a library-not-found error (*e.g.* HDF5 or FFTW),
   make sure the corresponding development headers and ``pkg-config`` files
   are on your ``PKG_CONFIG_PATH``, or pass the paths explicitly via
   ``CMAKE_PREFIX_PATH``:

   .. code-block:: bash

      CMAKE_PREFIX_PATH=/path/to/hdf5:/path/to/fftw pip install -e .

FFT backend and licensing
=========================

The Fortran core selects its FFT implementation at build time through the
``SPECTRE_FFT_BACKEND`` CMake option; the Python field-line tracer selects its
ODE integrator at run time (the ``rk_backend`` argument). By default the
fastest options are used: **FFTW3** (GPL) for the FFT and **CyRK**
(CC-BY-NC-SA, NonCommercial) for the tracer.

For commercial use, build with the ``SPECTRE_COMMERCIAL_USE`` environment
variable set. Both defaults then switch to permissively-licensed backends:
**pocketfft** (BSD-3-Clause, fetched at configure time, so FFTW is not needed)
and scipy's ``solve_ivp``:

.. code-block:: bash

   SPECTRE_COMMERCIAL_USE=1 pip install -e .
   pip uninstall CyRK   # optional: remove the NonCommercial package entirely

.. note::

   CyRK is a regular dependency, and pip has no mechanism to opt out of a
   dependency at install time — hence the explicit ``pip uninstall`` as the
   last step if the NonCommercial package must not be present at all. SPECTRE
   never imports CyRK unless the ``cyrk`` backend is actually used, so a
   CyRK-less installation is fully functional (the tracer uses scipy).
   Afterwards ``pip check`` will report the missing dependency; this is
   harmless, but reinstalling or upgrading ``spectre`` will pull CyRK back
   in, so repeat the uninstall after a rebuild.

A specific FFT backend (``fftw``, ``mkl``, or ``pocketfft``) can also be chosen
explicitly:

.. code-block:: bash

   pip install -e . -C cmake.define.SPECTRE_FFT_BACKEND=mkl

The choices compiled into an installation are recorded in
``spectre._build_info`` (``fft_backend`` and ``commercial_use``).

Platform-specific instructions
===============================

HPC cluster with a module system
--------------------------------

Load the required modules, create a virtual environment that inherits the
system MPI/HDF5 packages, then install SPECTRE.

.. rubric:: Jed (EPFL)

.. code-block:: bash

   module purge
   module load gcc openmpi openblas hdf5/1.14.3-mpi fftw/3.3.10-mpi-openmp
   module load python py-numpy py-pip py-mpi4py

   python -m venv --system-site-packages ~/virtualenvs/venv-spectre
   source ~/virtualenvs/venv-spectre/bin/activate

   git clone git@gitlab.com:spectre-eq/spectre.git
   cd spectre
   pip install -e .

.. rubric:: Pitagora (CINECA)

.. code-block:: bash

   module purge
   module load gcc
   module load openmpi/4.1.6--gcc--12.3.0-ucx1.20
   module load openblas/0.3.26--gcc--12.3.0
   module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.3.0-ucx1.20
   module load fftw/3.3.10--openmpi--4.1.6--gcc--12.3.0-ucx1.20
   module load python py-mpi4py/3.1.5--openmpi--4.1.6--gcc--12.3.0-ucx1.20

   python -m venv --system-site-packages ~/virtualenvs/venv-spectre
   source ~/virtualenvs/venv-spectre/bin/activate

   git clone git@gitlab.com:spectre-eq/spectre.git
   cd spectre
   pip install -e .

Note that module names may change as the system software stack evolves. Make
sure to update them accordingly if you encounter build errors related to missing
libraries.

Conda-based systems (SPC, local workstations)
---------------------------------------------

Use Conda to install the compiled libraries and the Fortran compiler, then
pip-install SPECTRE into the activated environment.

.. code-block:: bash

   conda create -n spectre_env
   conda activate spectre_env
   conda install python==3.13.2 pip gfortran_linux-64=11 libgfortran=14 \
       openblas mpi4py openmpi fftw hdf5

   # Exit and reopen the terminal so the new compiler is on PATH, then:
   conda activate spectre_env
   git clone git@gitlab.com:spectre-eq/spectre.git
   cd spectre
   pip install -e .

.. note::

   On some systems, conda installs MKL as a BLAS provider which can conflict
   with OpenBLAS.  If you encounter runtime errors related to BLAS, force
   OpenBLAS by setting:

   .. code-block:: bash

      export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

   before running SPECTRE.

Verifying the installation
==========================

Run the test suite from the root of the repository to confirm that the
Fortran compilation and Python bindings are working correctly:

.. code-block:: bash

   pytest

A successful run with no errors indicates that the installation is complete.

Optional: development dependencies
==================================

If you plan to add contributions to SPECTRE, you may want to install the
development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will include various tools for code formatting and linting.

Optional: documentation dependencies
====================================

To build this documentation locally, install the extra ``docs`` dependencies:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The rendered HTML will be available under ``docs/build/html/``.

Next steps
==========

Once the installation is verified, head to :doc:`generating_input` to prepare
your first equilibrium calculation.
