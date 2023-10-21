#!/bin/bash

source_dir=../../source/frame_distinction
data_dir=../../data/frame_type_count_estimation
input_dir=../../donwload/elmo

resources=(framenet propbank)

#model_name=elmo
#model_name=bert-base-uncased
model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

device=cuda:1

for resource in ${resources[@]}; do
    d1=${resource}
    d2=${model_name}
    python ${source_dir}/get_embeddings.py \
        --input_file ${data_dir}/dataset/${d1}/exemplars.jsonl \
        --elmo_dir ${input_dir} \
        --output_dir ${data_dir}/embeddings/${d1}/${d2} \
        --model_name ${model_name} \
        --device ${device} \
        --batch_size 32
done
