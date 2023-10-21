#!/bin/bash

source_dir=../../source/frame_type_count_estimation
data_dir=../../data/frame_type_count_estimation
input_dir=../data/frame_distinction

resource=framenet
# resource=propbank

model_name=bert-large-uncased

criterion=bic
#criterion=abic

d1=${resource}/${model_name}
d2=${criterion}

sets=dev
d3=${sets}
python ${source_dir}/verb_sense_clustering.py \
    --input_dir ${data_dir}/embeddings/${d1} \
    --input_dev_dir ${input_dir}/verb_sense_clustering/${d1}/dev \
    --output_dir ${data_dir}/frame_type_count_estimation/${d1}/${d2}/${d3} \
    --model_name ${model_name} \
    --layer best \
    --sets ${sets} \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical

sets=test
d3=${sets}
python ${source_dir}/verb_sense_clustering.py \
    --input_dir ${data_dir}/embeddings/${d1} \
    --input_dev_dir ${input_dir}/verb_sense_clustering/${d1}/dev \
    --output_dir ${data_dir}/frame_type_count_estimation/${d1}/${d2}/${d3} \
    --model_name ${model_name} \
    --layer best \
    --sets ${sets} \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical
