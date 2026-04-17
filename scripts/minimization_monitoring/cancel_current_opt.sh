#!/bin/sh

scancel --signal=SIGUSR1 $(cat slurmid)
