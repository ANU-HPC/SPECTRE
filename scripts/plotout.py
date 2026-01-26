#!/usr/bin/env python
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import sys
from re import compile, search

if len(sys.argv) < 2:
    print("Usage: python script.py <log_file>")
    sys.exit(1)

for i in range(1, len(sys.argv)):
    log_file = sys.argv[i]
    try:
        with open(log_file, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)

    VAR_PATTERNS = [
        (compile(r"iter\s+(\d+)"), "iter"),
        (compile(r"nfev\s+(\d+)"), "nfev"),
        (compile(r"cost\s+([\d\.eE+-]+)"), "cost"),
        (compile(r"<f>\s+([\d\.eE+-]+)"), "f_avg"),  # handles <f> notation
        (compile(r"fr\s+([\d\.eE+-]+)"), "f_real"),  # handles freal notation
        (compile(r"f_real\s+([\d\.eE+-]+)"), "f_real"),  # alternate spelling
        (compile(r"stpnrm\s+([\d\.eE+-]+)"), "stpnrm"),
        (compile(r"costred\s+([\d\.eE+-]+)"), "costred"),
        (compile(r"optim\s+([\d\.eE+-]+)"), "optim"),
    ]

    iterations, nfev, cost = [], [], []
    f_avg, f_real, stpnrm = [], [], []
    costred, optim = [], []

    for line in lines:
        if not line.strip() or not line.startswith("iter"):
            continue

        if search(r"iter\s+0\b", line):
            continue

        found_values = {}
        for pattern, var_name in VAR_PATTERNS:
            match = pattern.search(line)
            if match and var_name not in found_values:
                value = match.group(1)
                try:
                    if "." in value or "e" in value.lower():
                        found_values[var_name] = float(value)
                    else:
                        found_values[var_name] = int(value)
                except ValueError:
                    pass

        if "iter" in found_values:
            iterations.append(found_values["iter"])
        if "nfev" in found_values:
            nfev.append(found_values["nfev"])
        if "cost" in found_values:
            cost.append(found_values["cost"])
        if "f_avg" in found_values:
            f_avg.append(found_values["f_avg"])
        if "f_real" in found_values:
            f_real.append(found_values["f_real"])
        if "stpnrm" in found_values:
            stpnrm.append(found_values["stpnrm"])
        if "costred" in found_values:
            costred.append(found_values["costred"])
        if "optim" in found_values:
            optim.append(found_values["optim"])

    ms = 2.0
    label_fontsize = 11

    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True, num=log_file)

    if cost:
        axes[0].semilogy(cost, ms=ms, marker="o", color="b", label="Cost")
        axes[0].set_ylabel("Cost", fontsize=label_fontsize)

    if f_avg:
        axes[1].semilogy(f_avg, ms=ms, marker="o", color="g", label="<f>")
        axes[1].set_ylabel("<f>", fontsize=label_fontsize)

    if f_real:
        axes[2].semilogy(f_real, ms=ms, marker="o", color="r", label="freal")
        axes[2].set_ylabel("freal", fontsize=label_fontsize)

    if stpnrm:
        axes[3].semilogy(stpnrm, ms=ms, marker="o", color="c", label="Step Norm")
        axes[3].set_ylabel("Step Norm", fontsize=label_fontsize)

    if costred:
        axes[4].semilogy(costred, ms=ms, marker="o", color="m", label="Cost Reduction")
        axes[4].set_ylabel("Cost Reduction", fontsize=label_fontsize)

    if optim:
        axes[5].semilogy(optim, ms=ms, marker="o", color="k", label="Optim")
        axes[5].set_ylabel("Optim", fontsize=label_fontsize)

    axes[5].set_xlabel("Iteration", fontsize=13)

    axes[5].set_xlim(-2, len(iterations) + 2)

    axes[1].set_ylim(auto=True)

    plt.tight_layout()

plt.show()
