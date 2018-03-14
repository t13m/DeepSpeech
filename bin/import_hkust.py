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

import joblib

UNK_WORD = '<UNK>'
#SPH_BIN = "%s/tools/sph2pipe_v2.5/sph2pipe" % os.environ["EESEN_ROOT"]

def _process(convert_command):
    os.system(convert_command)
    return None

def prepare_data(data_dir, target_dir):
    target_wav_dir = os.path.join(target_dir, 'wav')
    if not os.path.exists(target_wav_dir):
        os.mkdir(target_wav_dir)

    wav_seg_path = os.path.join(data_dir, 'segments')
    with open(wav_seg_path) as fin:
        segment_lines = fin.readlines()
    segment_lines = [line.strip().split() for line in segment_lines if line.strip() != '']

    wav_scp_path = os.path.join(data_dir, 'wav.scp')
    with open(wav_scp_path) as scp_f:
        scp_lines = scp_f.readlines()
    wav_scp = [(line.split()[0], line.split()[1:-1])
           for line in scp_lines if line.strip() != '']
    wav_dict = dict(wav_scp)

    result = {}
    count = 0
    commands = []
    for seg in segment_lines:
        wav_filename = os.path.abspath("%s/%s.wav" % (target_wav_dir, seg[0]))
        cmd_part = '-t %s:%s' % (seg[-2], seg[-1])
        cmd_part_list = [wav_dict[seg[1]][0]] + [cmd_part] + wav_dict[seg[1]][1:] + [wav_filename]
        convert_command = ' '.join(cmd_part_list)
        commands.append(convert_command)
        result[seg[0]] = wav_filename
        count += 1
        if count % 100 == 0:
            print("%d/%d file processed." % (count, len(segment_lines)))

    joblib.Parallel(n_jobs=50)(joblib.delayed(_process)(cmd) for cmd in commands)

    return result

def prepare_trans(data_dir, lang_dir):
    dict_file = os.path.join(lang_dir, 'lexicon_numbers.txt')
    trans_file = os.path.join(data_dir, 'text')
    alphabet_file = os.path.join(lang_dir, 'units.txt')

    unk_word = UNK_WORD

    is_char = False
    if len(sys.argv) == 5:
        is_char = True
        space_word = ' '

    result = {}

    # read alphabet
    with open(alphabet_file) as fin:
        alphabet_lines = fin.readlines()
    alphabet = [reversed(line.strip().split()) for line in alphabet_lines if line.strip() != '']
    alphabet = dict(alphabet)

    # read the lexicon into a dictionary data structure
    fread = open(dict_file, 'r')
    lexicon_dict = {}
    for line in fread.readlines():
        line = line.replace('\n', '')
        splits = line.split(' ')  # assume there are no multiple spaces
        word = splits[0]
        letters = ''
        for n in range(1, len(splits)):
            letters += alphabet[splits[n]] + ' '
        lexicon_dict[word] = letters.strip()
    fread.close()

    # assume that each line is formatted as "uttid word1 word2 word3 ...", with no multiple spaces appearing
    fread = open(trans_file, 'r')
    for line in fread.readlines():
        out_line = ''
        line = line.replace('\n', '').strip()
        while '  ' in line:
            line = line.replace('  ', ' ')  # remove multiple spaces in the transcripts

        uttid = line.split(' ')[0]  # the first field is always utterance id
        trans = line.replace(uttid, '').strip()
        if is_char:
            trans = trans.replace(' ', ' ' + space_word + ' ')
        splits = trans.split(' ')

        #out_line += uttid + ' '
        for n in range(0, len(splits)):
            try:
                out_line += lexicon_dict[splits[n]] + ' '
            except Exception:
                out_line += lexicon_dict[unk_word] + ' '

        result[uttid] = out_line.strip()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import HKUST corpus into CSV files")
    parser.add_argument(
        '--train-data-dir',
        help='HKUST train data directory, eg. \'/path/to/data/local/train\''
    )
    parser.add_argument(
        '--dev-data-dir',
        help='HKUST dev data directory, eg. \'/path/to/data/local/dev\''
    )
    parser.add_argument(
        '--lang-dir',
        help='lang directory (lexicon_numbers.txt, tokens.txt, units.txt, words.txt)'
    )
    parser.add_argument(
        '--target-dir',
        help='Directory for generated files'
    )
    parser.print_help()

    args = parser.parse_args()
    if args.dev_data_dir and args.lang_dir and args.target_dir:
        print("Processing devset")
        if not os.path.exists(args.target_dir):
            os.mkdir(args.target_dir)
        if not os.path.exists(args.target_dir + '/dev'):
            os.mkdir(args.target_dir + '/dev')
        wav_part = prepare_data(args.dev_data_dir, args.target_dir + '/dev')
        trans_part = prepare_trans(args.dev_data_dir, args.lang_dir)
        files = [(wav_part[key], os.path.getsize(wav_part[key]), trans_part[key]) for key in trans_part.keys()]
        devset = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])
        devset.to_csv(os.path.join(args.target_dir, 'hkust_phn_dev.csv'), index=False)

    if args.train_data_dir and args.lang_dir and args.target_dir:
        print("Processing trainset")
        if not os.path.exists(args.target_dir):
            os.mkdir(args.target_dir)
        if not os.path.exists(args.target_dir + '/train'):
            os.mkdir(args.target_dir + '/train')
        wav_part = prepare_data(args.train_data_dir, args.target_dir + '/train')
        trans_part = prepare_trans(args.train_data_dir, args.lang_dir)
        files = [(wav_part[key], os.path.getsize(wav_part[key]), trans_part[key]) for key in trans_part.keys()]
        trainset = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])
        trainset.to_csv(os.path.join(args.target_dir, 'hkust_phn_train.csv'), index=False)
