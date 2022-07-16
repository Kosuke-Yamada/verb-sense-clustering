#!/bin/sh

model_name=all-in-one-cluster
#model_name=elmo
#model_name=bert-base-uncased
#model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

#sets=dev
sets=test

python aggregate_relations.py \
    --dataset_path ../data/experiment_frame_distinction/dataset/framenet \
    --score_path ../data/experiment_frame_distinction/frame_distinction/framenet \
    --f2f_path ../data/preprocessing/relations \
    --output_path ../data/experiment_frame_distinction/relations \
    --model_name ${model_name} \
    --layer -1 \
    --sets ${sets}
