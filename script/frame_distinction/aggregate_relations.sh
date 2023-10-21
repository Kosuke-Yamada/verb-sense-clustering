#!/bin/sh

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction
input_dir=../../data/preprocessing

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

python ${source_dir}/aggregate_relations.py \
    --dataset_path ${data_dir}/dataset/framenet \
    --score_path ${data_dir}/frame_distinction/framenet \
    --f2f_path ${input_dir}/relations \
    --output_path ${data_dir}/relations \
    --model_name ${model_name} \
    --layer -1 \
    --sets ${sets}
