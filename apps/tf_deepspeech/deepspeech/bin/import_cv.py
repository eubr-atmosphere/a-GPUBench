#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import tarfile
import requests
import subprocess
from glob import glob
from os import path
from sox import Transformer
from threading import Lock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.progress import print_progress

from os import walk

import argparse

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
SAMPLE_RATE = 16000
MAX_SECS = 10
ARCHIVE_DIR_NAME = 'cv_corpus_v1'
ARCHIVE_NAME = ARCHIVE_DIR_NAME + '.tar.gz'
ARCHIVE_URL = 'https://s3.us-east-2.amazonaws.com/common-voice-data-download/' + ARCHIVE_NAME

TRAIN_DIR_NAME = 'cv-valid-train'
TEST_DIR_NAME = 'cv-valid-test'
DEV_DIR_NAME = 'cv-valid-dev'

TRAIN_CSV_NAME = 'cv-valid-train'
DEV_CSV_NAME = 'cv-valid-dev'
TEST_CSV_NAME = 'cv-valid-test'

MAX_TRAIN = 10
MAX_TEST = 5
MAX_DEV = 5


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Download Common Voice data and extract features")
    parser.add_argument(
        "--traindirname",
        help="Specify train dir name",
        default='cv-valid-train'
    )
    parser.add_argument(
        "--devdirname",
        help="Specify dev dir name",
        default='cv-valid-dev'
    )
    parser.add_argument(
        "--testdirname",
        help="Specify test dir name",
        default='cv-valid-test'
    )
    parser.add_argument(
        "--csvtrain",
        help="Specify csv train name",
        default='cv-valid-train'
    )
    parser.add_argument(
        "--csvdev",
        help="Specify csv dev name",
        default='cv-valid-dev'
    )
    parser.add_argument(
        "--csvtest",
        help="Specify csv test name",
        default='cv-valid-test'
    )
    parser.add_argument(
        "--maxtrain",
        help="Specify max number of train files",
        default=10
    )
    parser.add_argument(
        "--maxdev",
        help="Specify max number of dev files",
        default=5
    )
    parser.add_argument(
        "--maxtest",
        help="Specify max number of test files",
        default=5
    )
    print("Import cv target dir: " + str(sys.argv[1]))
    print("Import cv arguments: "+ str(sys.argv[2:]))
    parsed_args = parser.parse_args(args=sys.argv[2:])
    print("Current import configuration: " + str(parsed_args))
    return parsed_args


def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = path.abspath(target_dir)
    # Conditionally download data
    archive_path = _maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
    # Conditionally extract common voice data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Conditionally convert common voice CSV files and mp3 data to DeepSpeech CSVs and wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)


def _maybe_download(archive_name, target_dir, archive_url):
    # If archive file does not exist, download it...
    archive_path = path.join(target_dir, archive_name)
    if not path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        req = requests.get(archive_url, stream=True)
        total_size = int(req.headers.get('content-length', 0))
        done = 0
        with open(archive_path, 'wb') as f:
            for data in req.iter_content(1024 * 1024):
                done += len(data)
                f.write(data)
                print_progress(done, total_size)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path


def _maybe_extract(target_dir, extracted_data, archive_path):
    tot_train = 0
    tot_test = 0
    tot_dev = 0
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = path.join(target_dir, extracted_data)
    if not path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % archive_path)
        with tarfile.open(archive_path) as tar:
            members = list(tar.getmembers())
            for i, member in enumerate(members):
                if (TRAIN_DIR_NAME in member.name) and (tot_train < MAX_TRAIN) and ('mp3' in member.name):
                    tar.extract(member, path=target_dir)
                    tot_train += 1
                if (DEV_DIR_NAME in member.name) and (tot_dev < MAX_DEV) and ('mp3' in member.name):
                    tar.extract(member, path=target_dir)
                    tot_dev += 1
                if (TEST_DIR_NAME in member.name) and (tot_test < MAX_TEST) and ('mp3' in member.name):
                    tar.extract(member, path=target_dir)
                    tot_test += 1
                if 'csv' in member.name:
                    tar.extract(member, path=target_dir)
    else:
        print('Found directory "%s" - not extracting it from archive.' % archive_path)


def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = path.join(target_dir, extracted_data)
    for source_csv in glob(path.join(extracted_dir, '*.csv')):
        if TRAIN_CSV_NAME in source_csv or DEV_CSV_NAME in source_csv or TEST_CSV_NAME in source_csv:
            _maybe_convert_set(extracted_dir, source_csv, path.join(target_dir, os.path.split(source_csv)[-1]))


def _maybe_convert_set(extracted_dir, source_csv, target_csv):
    print()
    if path.exists(target_csv):
        print('Found CSV file "%s" - not importing "%s".' % (target_csv, source_csv))
        return
    print('No CSV file "%s" - importing "%s"...' % (target_csv, source_csv))

    train_dir = path.join(extracted_dir, TRAIN_DIR_NAME)
    dev_dir = path.join(extracted_dir, DEV_DIR_NAME)
    test_dir = path.join(extracted_dir, TEST_DIR_NAME)

    train_files = glob(path.join(train_dir, '*.mp3'))
    dev_files = glob(path.join(dev_dir, '*.mp3'))
    test_files = glob(path.join(test_dir, '*.mp3'))

    samples = []
    with open(source_csv) as source_csv_file:
        reader = csv.DictReader(source_csv_file)
        for row in reader:

            if (((TRAIN_CSV_NAME in source_csv) and any(
                    str(row['filename']) in train_file for train_file in train_files)) or
                    ((DEV_CSV_NAME in source_csv) and any(
                        str(row['filename']) in dev_file for dev_file in dev_files)) or
                    ((TEST_CSV_NAME in source_csv) and any(
                        str(row['filename']) in test_file for test_file in test_files))):
                samples.append((row['filename'], row['text']))

    # Mutable counters for the concurrent embedded routine
    counter = {'all': 0, 'too_short': 0, 'too_long': 0}
    lock = Lock()
    num_samples = len(samples)
    rows = []

    def one_sample(sample):
        mp3_filename = path.join(*(sample[0].split('/')))
        mp3_filename = path.join(extracted_dir, mp3_filename)
        # Storing wav files next to the mp3 ones - just with a different suffix
        wav_filename = path.splitext(mp3_filename)[0] + ".wav"
        _maybe_convert_wav(mp3_filename, wav_filename)
        frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        file_size = path.getsize(wav_filename)
        with lock:
            if int(frames / SAMPLE_RATE * 1000 / 10 / 2) < len(str(sample[1])):
                # Excluding samples that are too short to fit the transcript
                counter['too_short'] += 1
            elif frames / SAMPLE_RATE > MAX_SECS:
                # Excluding very long samples to keep a reasonable batch-size
                counter['too_long'] += 1
            else:
                # This one is good - keep it for the target CSV
                rows.append((wav_filename, file_size, sample[1]))
            print_progress(counter['all'], num_samples)
            counter['all'] += 1

    print('Importing mp3 files...')
    pool = Pool(cpu_count())
    pool.map(one_sample, samples)
    pool.close()
    pool.join()

    print_progress(num_samples, num_samples)

    print('Writing "%s"...' % target_csv)
    with open(target_csv, 'w') as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, row in enumerate(rows):
            filename, file_size, transcript = row
            print_progress(i + 1, len(rows))
            writer.writerow({'wav_filename': filename, 'wav_filesize': file_size, 'transcript': transcript})

    print('Imported %d samples.' % (counter['all'] - counter['too_short'] - counter['too_long']))
    if counter['too_short'] > 0:
        print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
    if counter['too_long'] > 0:
        print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], MAX_SECS))


def _maybe_convert_wav(mp3_filename, wav_filename):
    if not path.exists(wav_filename):
        transformer = Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        transformer.build(mp3_filename, wav_filename)


if __name__ == "__main__":
    args = _parse_arguments()
    TRAIN_DIR_NAME = args.traindirname
    DEV_DIR_NAME = args.devdirname
    TEST_DIR_NAME = args.testdirname
    TRAIN_CSV_NAME = args.csvtrain
    DEV_CSV_NAME = args.csvdev
    TEST_CSV_NAME = args.csvtest
    MAX_TRAIN = int(args.maxtrain)
    MAX_DEV = int(args.maxdev)
    MAX_TEST = int(args.maxtest)

    print("traindirname: " + TRAIN_DIR_NAME + " \n" 
          "devdirname: " + DEV_DIR_NAME + " \n" 
          "testdirname: " + TEST_DIR_NAME + " \n" 
          "train_csv: " + TRAIN_CSV_NAME + " \n" 
          "dev_csv: " + DEV_CSV_NAME + " \n" 
          "test_csv: " + TEST_CSV_NAME + " \n" 
          "max_train: " + str(MAX_TRAIN) + " \n" 
          "max_dev: " + str(MAX_DEV) + " \n" 
          "max_test: " + str(MAX_TEST)
          )
    __target_dir = sys.argv[1]

    print("INFO Args: " + str(args))

    _download_and_preprocess_data(__target_dir)
