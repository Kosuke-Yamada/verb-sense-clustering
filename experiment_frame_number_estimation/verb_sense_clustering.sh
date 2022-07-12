#!/bin/sh

model_name=bert-large-uncased

#resource=framenet
resource=propbank

criterion=bic
#criterion=abic

python verb_sense_clustering.py \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets dev \
    --criterion ${criterion}

python verb_sense_clustering.py \
    --model_name ${model_name} \
    --resource ${resource} \
    --layer -1 \
    --sets test \
    --criterion ${criterion}
