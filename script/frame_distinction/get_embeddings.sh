#!/bin/sh

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction
input_dir=../../donwload/elmo

resource=framenet
#resource=propbank

device=cuda:0

model_name=elmo
#model_name=bert-base-uncased
#model_name=bert-large-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
#model_name=xlnet-base-cased

python ../experiment_frame_distinction/get_embeddings.py \
    --input_path ${data_dir}/dataset \
    --elmo_path ${input_dir} \
    --output_path ${data_dir}/embeddings \
    --model_name ${model_name} \
    --device ${device} \
    --resource ${resource} \
    --batch_size 32
