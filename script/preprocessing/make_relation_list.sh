#!/bin/sh

source_dir=../../source/prprocessing
data_dir=../../data/prprocessing
input_dir=../../download/fndata-1.7

python ${source_dir}/make_relation_list.py \
    --input_path ${input_dir}/frame \
    --output_path ${data_dir}/relations
