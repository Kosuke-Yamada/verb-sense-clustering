#!/bin/sh

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
    --input_path ../data/experiment_frame_distinction/dataset \
    --elmo_path ../data/raw/elmo \
    --output_path ../data/experiment_frame_distinction/embeddings \
    --model_name ${model_name} \
    --device ${device} \
    --resource ${resource} \
    --batch_size 32
