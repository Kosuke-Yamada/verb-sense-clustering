#!/bin/sh

model_name=bert-large-uncased

#resource=framenet
resource=propbank

criterion=bic
#criterion=abic

python verb_sense_clustering.py \
    --input_path ../data/experiment_frame_number_estimation/embeddings \
    --dev_path ../data/experiment_frame_distinction/frame_distinction \
    --output_path ../data/experiment_frame_number_estimation/frame_number_estimation \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets dev \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical

python verb_sense_clustering.py \
    --input_path ../data/experiment_frame_number_estimation/embeddings \
    --dev_path ../data/experiment_frame_distinction/frame_distinction \
    --output_path ../data/experiment_frame_number_estimation/frame_number_estimation \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets test \
    --criterion ${criterion} \
    --n_seeds 5 \
    --covariance_type spherical
