# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import sys
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from inspect import getsource
from tempfile import NamedTemporaryFile
from pathlib import Path
from importlib import import_module, invalidate_caches


def run_pool(run_func, files, max_workers=4):
    outputs = []

    with (
        ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool,
        enable_spawn(run_func) as target,
    ):
        futures = [pool.submit(target, files[i]) for i in range(len(files))]
        for future in as_completed(futures):
            outputs.append(future.result())

    return outputs


@contextmanager
def enable_spawn(func):
    invalidate_caches()
    source = getsource(func)
    with NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write(source)
        f.flush()
        path = Path(f.name)
        directory = str(path.parent)
        sys.path.append(directory)
        module = import_module(str(path.stem))
        yield getattr(module, func.__name__)
        sys.path.remove(directory)
