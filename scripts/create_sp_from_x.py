#!/usr/bin/env python
from argparse import ArgumentParser

parser = ArgumentParser(
    description='Plot SPECTRE interfaces from configuration file and data files'
)
parser.add_argument('config_file',
                    help='SPECTRE configuration file')
parser.add_argument('data_file',
                    help='Npy file with the interface information')
args = parser.parse_args()

from numpy import load
from spectre import SPECTRE, force_real, get_freal_fsa_metric

test = SPECTRE.from_input_file(args.config_file, verbose=False)

x = load(args.data_file)

force_real(x, test)

fsa = get_freal_fsa_metric(test)
print(f"<f>  {fsa:.4e}")

test.allglobal.wrtend()
