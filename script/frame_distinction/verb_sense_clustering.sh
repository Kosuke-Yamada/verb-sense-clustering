#!/bin/sh

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction

resource=framenet
#resource=propbank

# model_name=bert-base-uncased
#model_name=albert-base-v2
#model_name=roberta-base
model_name=gpt2
#model_name=xlnet-base-cased
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12)

# model_name=all-in-one-cluster
# layers=(0)

# model_name=elmo
# layers=(0 1 2)

# model_name=bert-large-uncased
# layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

for resource in framenet propbank; do
    for layer in ${layers[@]}; do
        python ${source_dir}/verb_sense_clustering.py \
            --input_path ${data_dir}/embeddings \
            --output_path ${data_dir}/frame_distinction \
            --resource ${resource} \
            --model_name ${model_name} \
            --layer ${layer} \
            --n_seed 5 \
            --covariance_type spherical \
            --sets dev
    done

    layer=-1
    python ${source_dir}/verb_sense_clustering.py \
        --input_path ${data_dir}/embeddings \
        --output_path ${data_dir}/frame_distinction \
        --resource ${resource} \
        --model_name ${model_name} \
        --layer ${layer} \
        --n_seed 5 \
        --covariance_type spherical \
        --sets test
done
