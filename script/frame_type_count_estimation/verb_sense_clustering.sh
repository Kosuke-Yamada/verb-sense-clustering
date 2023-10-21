#!/bin/sh

source_dir=../../source/frame_type_count_estimation
data_dir=../../data/frame_type_count_estimation

model_name=bert-large-uncased

#resource=framenet
resource=propbank

criterion=bic
#criterion=abic

python ${source_dir}/verb_sense_clustering.py \
    --input_path ${data_dir}/embeddings \
    --dev_path ../data/experiment_frame_distinction/frame_distinction \
    --output_path ${data_dir}/frame_number_estimation \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets dev \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical

python ${source_dir}/verb_sense_clustering.py \
    --input_path ${data_dir}/embeddings \
    --dev_path ../data/experiment_frame_distinction/frame_distinction \
    --output_path ${data_dir}/frame_number_estimation \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets test \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical
