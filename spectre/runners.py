# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from subprocess import Popen, PIPE
from textwrap import dedent
from inspect import getsource


def run_slurm(setup_commands, name, ext, run):
    sourcefile = getsource(run)
    s = "def run():\n"
    sourcefile = sourcefile.replace(s, s + f"\n\text = '{ext}'\n\tname = '{name}'\n") + "run()\n"
    with open(f".tmpcode_{name}", "w") as f:
        f.writelines(sourcefile)

    slurm_commands = (
        "#!/bin/sh \n#SBATCH --signal=SIGUSR1@180\n"
        + dedent(setup_commands)
        + dedent(
            f"""
            cp .tmpcode_{name} out_{name}.txt

            #export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
            export PYTHONUNBUFFERED=1
            
            #export UCX_TLS=self,sm,ud
            # export UCX_UD_MLX5_RX_QUEUE_LEN=16384
            # export UCX_TLS=rc,dc,ud,sysv,posix,self # To be checked, nope not worrking

            srun python .tmpcode_{name}

            ## for profiling the python side
            #srun pyinstrument -r html -o timing_{name}.html  .tmpcode_{name}
            #srun python -m cProfile -o timing_{name}.cprof .tmpcode_{name}

            rm slurmid
        """
        )
    )

    with open(f".tmpsbatch_{name}", "w") as f:
        print(slurm_commands, file=f)

    proc = Popen(f"sbatch -W .tmpsbatch_{name}", shell=True, stdout=PIPE)

    with open("slurmid", "w") as f:
        print(str(proc.stdout.readline())[:-3].split(" ")[-1], file=f)


def run_local(setup_commands, name, ext, run, num_cpus, num_threads=1):
    sourcefile = getsource(run)
    s = "def run():\n"
    sourcefile = sourcefile.replace(s, s + f"\n\text = '{ext}'\n\tname = '{name}'\n") + "run()\n"
    with open(f".tmpcode_{name}", "w") as f:
        f.writelines(sourcefile)
    
    run_commands = (
        "#!/bin/sh\n"
        + dedent(setup_commands)
        + dedent(f"""\n\
    export OMP_NUM_THREADS={num_threads}\n
    rank=`python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())'`
    if [ ${{rank}} -eq 0 ]; then cp .tmpcode_{name} out_{name}.txt; fi\n
    python -u .tmpcode_{name} | tee -a out_{name}.txt\n""")
    )

    with open(f"run_forcemin_{name}.sh", "w") as f:
        f.writelines(run_commands)
