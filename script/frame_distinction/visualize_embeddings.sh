#!/bin/sh

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction

resource=framenet
#resource=propbank

#model_name=all-in-one-cluster
#model_name=elmo
#model_name=bert-base-uncased
model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

#sets=dev
sets=test

python ${source_dir}/visualize_embeddings.py \
    --input_path ${data_dir}/embeddings \
    --dev_path ${data_dir}/frame_distinction \
    --output_path ${data_dir}/visualization \
    --resource ${resource} \
    --model_name ${model_name} \
    --layer -1 \
    --n_seeds 5 \
    --covariance_type spherical \
    --sets ${sets} \
    --random_state 0 \
    --verb all_verbs
