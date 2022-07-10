#!/bin/sh

for model_name in all-in-one-cluster elmo bert-base-uncased bert-large-uncased roberta-base albert-base-v2 gpt2 xlnet-base-cased; do
    python aggregate_relations.py --model_name ${model_name}
done
