#!/bin/bash

source_dir=../../source/preprocessing
data_dir=../../data/preprocessing
input_dir=../../download/fndata-1.7

python ${source_dir}/make_relations.py \
    --input_dir ${input_dir}/frame \
    --output_dir ${data_dir}/relations
