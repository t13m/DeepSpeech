# encoding=utf-8
#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import fnmatch
import pandas
import subprocess
import unicodedata
import wave
import codecs
import numpy as np
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import HKUST corpus into CSV files")
    parser.add_argument(
        '--csv',
        help='csv file that you want to count'
    )
    parser.add_argument(
        '--units',
        help='token id'
    )
    parser.add_argument(
        '--output',
        help='output path'
    )
    args = parser.parse_args()
    with open(args.units) as fin:
        lines = fin.readlines()
    lines = [line.strip().split() for line in lines if line.strip() != '']

    # EESEN 认为0元素为blank
    # TF 认为last元素为blank
    # 此处遵照EESEN格式
    alphabet = dict(lines)
    counts = np.zeros([len(alphabet) + 1], dtype=np.int64)

    file = pandas.read_csv(args.csv, encoding='utf-8', na_filter=False)
    trans = file.ix[:, ["transcript"]].values
    for sentence in trans:
        counts[0] += 1
        for part in sentence[0].split():
            counts[int(alphabet[part])] += 1
            counts[0] += 1

    np.savetxt(args.output, counts, fmt='%i')