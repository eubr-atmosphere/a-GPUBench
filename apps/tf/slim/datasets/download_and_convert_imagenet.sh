#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
# Copyright 2021 Giovanni Dispoto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download and preprocess ImageNet Challenge 2012
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-00127-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  cd research/slim
#  bazel build :download_and_convert_imagenet
#  ./bazel-bin/download_and_convert_imagenet.sh [data-dir]
set -e

if [ -z "$1" ]; then
  echo "usage download_and_convert_imagenet.sh [data dir]"
  exit
fi

# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/raw-data/"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"
WORK_DIR="$0.runfiles/__main__"

# Download the ImageNet data.
LABELS_FILE="imagenet_lsvrc_2015_synsets.txt"
DOWNLOAD_SCRIPT="./download_imagenet.sh"
"${DOWNLOAD_SCRIPT}" "${SCRATCH_DIR}" "${LABELS_FILE}"

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}validation/"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="imagenet_2012_validation_synset_labels.txt"

"python3" "${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="process_bounding_boxes.py"
BOUNDING_BOX_FILE="${SCRATCH_DIR}imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${SCRATCH_DIR}bounding_boxes/"

"python3" "${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
# | sort >"${BOUNDING_BOX_FILE}"
echo "Finished downloading and preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="build_imagenet_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}"
IMAGENET_METADATA_FILE="imagenet_metadata.txt"

#move outside the current directory in order to call the script inside the right enviroment
cd ../../../../

#NV_GPU=2 nvidia-docker run --user 1060:1060 -v ~/storage/data/:/data dispoto /bin/bash -c ` 
python3.7 -m pipenv run python3.7 ./apps/tf/slim/datasets/"${BUILD_SCRIPT}" \
  --train_directory="/home/Desktop/output/imagenet/raw-data/train" \
  --validation_directory="/home/desktop/output/dataset/imagenet/raw-data/val" \
  --output_directory="/home/Desktop/output/dataset/imagenet/" \
  --imagenet_metadata_file="apps/tf/slim/datasets/${IMAGENET_METADATA_FILE}" \
  --labels_file="apps/tf/slim/datasets/${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"
#  `