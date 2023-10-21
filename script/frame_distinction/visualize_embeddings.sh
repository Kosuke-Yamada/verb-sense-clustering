#!/bin/bash

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction

resources=(framenet propbank)

#model_name=all-in-one-cluster
model_name=elmo
#model_name=bert-base-uncased
# model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

#sets=dev
sets=test

for resource in ${resources[@]}; do
    d1=${resource}/${model_name}
    d2=${sets}
    python ${source_dir}/visualize_embeddings.py \
        --input_dir ${data_dir}/embeddings/${d1} \
        --input_dev_dir ${data_dir}/verb_sense_clustering/${d1}/dev \
        --output_dir ${data_dir}/visualization/${d1}/${d2} \
        --resource ${resource} \
        --model_name ${model_name} \
        --layer best \
        --n_seeds 5 \
        --covariance_type spherical \
        --sets ${sets} \
        --random_state 0 \
        --verb all_verbs
done
