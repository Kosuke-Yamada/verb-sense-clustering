#!/bin/sh

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

python get_embeddings.py \
    --model_name ${model_name} \
    --device ${device} \
    --resource ${resource}
