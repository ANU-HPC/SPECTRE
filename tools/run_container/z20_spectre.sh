#!/bin/bash
# Activate the SPECTRE Python environment and add all scripts subdirectories to
# PATH so run scripts are directly callable from any shell session.
source /opt/spectre-env/bin/activate

export SPECTRE_ROOT="/opt/spectre"

for d in /opt/spectre/scripts/*/; do
    case ":$PATH:" in
        *":$d:"*) ;;
        *) export PATH="$d:$PATH" ;;
    esac
done
