#!/bin/sh

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
        python ../experiment_frame_distinction/verb_sense_clustering.py \
            --input_path ../data/experiment_frame_distinction/embeddings \
            --output_path ../data/experiment_frame_distinction/frame_distinction \
            --resource ${resource} \
            --model_name ${model_name} \
            --layer ${layer} \
            --n_seed 5 \
            --covariance_type spherical \
            --sets dev
    done

    layer=-1
    python ../experiment_frame_distinction/verb_sense_clustering.py \
        --input_path ../data/experiment_frame_distinction/embeddings \
        --output_path ../data/experiment_frame_distinction/frame_distinction \
        --resource ${resource} \
        --model_name ${model_name} \
        --layer ${layer} \
        --n_seed 5 \
        --covariance_type spherical \
        --sets test
done
