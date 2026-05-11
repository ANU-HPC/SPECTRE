#!/bin/bash
source /etc/profile.d/z10_spack_environment.sh
source /etc/profile.d/z20_spectre.sh
exec "$@"
