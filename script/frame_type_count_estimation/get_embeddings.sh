#!/bin/sh

source_dir=../../source/frame_type_count_estimation
data_dir=../../data/frame_type_count_estimation
input_dir=../../donwload/elmo

device=cuda:1

#model_name=elmo
#model_name=bert-base-uncased
model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

#resource=framenet
resource=propbank

python ${source_dir}/get_embeddings.py \
    --input_path ${data_dir}/dataset \
    --elmo_path ${input_dir} \
    --output_path ${data_dir}/embeddings \
    --model_name ${model_name} \
    --device ${device} \
    --resource ${resource} \
    --batch_size 32
