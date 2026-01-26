#!/bin/sh

tail -f -n +1 $1 | grep --line-buffered "iter  "
