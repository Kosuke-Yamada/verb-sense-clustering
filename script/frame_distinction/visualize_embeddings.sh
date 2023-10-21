#!/bin/sh

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

python ../experiment_frame_distinction/visualize_embeddings.py \
    --input_path ../data/experiment_frame_distinction/embeddings \
    --dev_path ../data/experiment_frame_distinction/frame_distinction \
    --output_path ../data/experiment_frame_distinction/visualization \
    --resource ${resource} \
    --model_name ${model_name} \
    --layer -1 \
    --n_seeds 5 \
    --covariance_type spherical \
    --sets ${sets} \
    --random_state 0 \
    --verb all_verbs
