#!/bin/sh

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

scancel --signal=SIGUSR1 $(cat slurmid)
