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
| **FFTW3**          | Only needed for ``SPECTRE_FFT_BACKEND=fftw``  |
|                    | builds (the default backend is pocketfft,     |
|                    | fetched at configure time; see below)         |
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
``SPECTRE_FFT_BACKEND`` CMake option:

* ``pocketfft`` (default) — BSD-3-Clause, fetched at configure time, so no
  system FFT library is needed. A default build is fully permissively
  licensed.
* ``fftw`` — FFTW3, GPL; the fastest option.
* ``mkl`` — Intel oneMKL's FFTW3 interface; proprietary but redistributable.

.. code-block:: bash

   pip install -e . -C cmake.define.SPECTRE_FFT_BACKEND=fftw

The Python field-line tracer integrates with **CyRK** (Apache-2.0).

.. note::

   CyRK changed its license from CC-BY-NC-SA-4.0 (NonCommercial) to
   Apache-2.0 in v0.16.1, but the PyPI artifacts up to and including 0.17.1
   still carry the old license string in their metadata. SPECTRE therefore
   installs CyRK straight from its GitHub repository, pinned to a commit
   where the license file and the package metadata both state Apache-2.0. A
   copy of CyRK's license is kept in ``thirdparty_licenses/`` and must
   accompany any distribution that bundles CyRK (the Docker run image does
   this).

SPECTRE's least-squares optimizers (``spectre/least_squares/``) and its
numerical-differentiation helpers (``spectre/numdiff_scipy.py``) are adapted
from **SciPy** (BSD-3-Clause). Because that code is part of SPECTRE's own
sources rather than an installed dependency, SciPy's license travels with
every copy of SPECTRE, including the wheel; the text is in
``thirdparty_licenses/scipy.LICENSE.txt``.

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
