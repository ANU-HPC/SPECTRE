#!/bin/sh

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

tail -f -n +1 $1 | grep --line-buffered "^iter  "
