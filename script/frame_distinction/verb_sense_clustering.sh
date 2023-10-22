#!/bin/bash

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction

resources=(framenet propbank)

# model_name=all-in-one-cluster
# layers=(00)

# model_name=elmo
# layers=(00 01 02)

# model_name=bert-base-uncased
# model_name=albert-base-v2
# model_name=roberta-base
# model_name=gpt2
model_name=xlnet-base-cased
layers=(00 01 02 03 04 05 06 07 08 09 10 11 12)

# model_name=bert-large-uncased
# layers=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

for resource in ${resources[@]}; do
    for layer in ${layers[@]}; do
        d1=${resource}/${model_name}
        python ${source_dir}/verb_sense_clustering.py \
            --input_dir ${data_dir}/embeddings/${d1} \
            --output_dir ${data_dir}/verb_sense_clustering/${d1}/dev \
            --model_name ${model_name} \
            --layer ${layer} \
            --n_seed 5 \
            --covariance_type spherical \
            --sets dev
    done

    d1=${resource}/${model_name}
    layer=best
    python ${source_dir}/verb_sense_clustering.py \
        --input_dir ${data_dir}/embeddings/${d1} \
        --output_dir ${data_dir}/verb_sense_clustering/${d1}/test \
        --input_dev_dir ${data_dir}/verb_sense_clustering/${d1}/dev \
        --model_name ${model_name} \
        --layer ${layer} \
        --n_seed 5 \
        --covariance_type spherical \
        --sets test
done
