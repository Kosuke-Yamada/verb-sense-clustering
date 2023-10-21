#!/bin/bash

source_dir=../../source/preprocessing
data_dir=../../data/preprocessing
input_dir=../../download/fndata-1.7

python ${source_dir}/extract_exemplars_framenet.py \
    --input_dir ${input_dir}/lu \
    --output_dir ${data_dir}/framenet
