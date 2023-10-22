#!/bin/bash

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction
input_dir=../../data/preprocessing

# model_name=all-in-one-cluster
# model_name=elmo
# model_name=bert-base-uncased
# model_name=bert-large-uncased
# model_name=albert-base-v2
# model_name=roberta-base
# model_name=gpt2
model_name=xlnet-base-cased

layer=best

#sets=dev
sets=test

d1=${model_name}/${sets}
python ${source_dir}/aggregate_relations.py \
    --input_file ${data_dir}/dataset/framenet/exemplars.jsonl \
    --input_score_file ${data_dir}/verb_sense_clustering/framenet/${d1}/verb_scores-${layer}.jsonl \
    --input_f2f_file ${input_dir}/relations/f2f_list.jsonl \
    --output_dir ${data_dir}/relations/${d1} \
    --model_name ${model_name} \
    --layer ${layer} \
    --sets ${sets}
