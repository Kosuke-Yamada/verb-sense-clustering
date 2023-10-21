#!/bin/bash

source_dir=../../source/preprocessing
data_dir=../../data/preprocessing
input_dir=../../download/ontonotes

python ${source_dir}/extract_exemplars_propbank.py \
    --input_frame_dir ${input_dir}/metadata/frames \
    --input_annotation_dir ${input_dir}/annotations \
    --output_dir ${data_dir}/propbank
