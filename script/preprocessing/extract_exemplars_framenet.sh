#!/bin/sh

source_dir=../../source/prprocessing
data_dir=../../data/prprocessing
input_dir=../../download/fndata-1.7

python ${source_dir}/extract_exemplars_framenet.py \
    --input_path ${input_dir}/lu \
    --output_path ${data_dir}/framenet
