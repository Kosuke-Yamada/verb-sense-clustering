#!/bin/sh

source_dir=../../source/prprocessing
data_dir=../../data/prprocessing
input_dir=../../download/ontonotes

python ${source_dir}/extract_exemplars_propbank.py \
    --input_frame_path ${input_dir}/metadata/frames \
    --input_anno_path ${input_dir}/annotations \
    --output_path ${data_dir}/propbank
