#!/bin/sh

# model_name=elmo
# for resource in framenet propbank; do
#     for layer in 0 1 2; do
#         python verb_sense_clustering.py \
#             --model_name ${model_name} \
#             --resource ${resource} \
#             --layer ${layer} \
#             --sets dev
#     done
#     python verb_sense_clustering.py \
#         --model_name ${model_name} \
#         --resource ${resource} \
#         --layer -1 \
#         --sets test
# done

# model_name=bert-large-uncased
# for resource in framenet propbank; do
#     for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
#         python verb_sense_clustering.py \
#             --model_name ${model_name} \
#             --resource ${resource} \
#             --layer ${layer}
#     done

#     python verb_sense_clustering.py \
#         --model_name ${model_name} \
#         --resource ${resource} \
#         --layer -1 \
#         --sets test
# done

#model_name=bert-base-uncased
#model_name=albert-base-v2
#model_name=roberta-base
#model_name=gpt2
model_name=xlnet-base-cased
for resource in framenet propbank; do
    for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
        python verb_sense_clustering.py \
            --model_name ${model_name} \
            --resource ${resource} \
            --layer ${layer}
    done

    python verb_sense_clustering.py \
        --model_name ${model_name} \
        --resource ${resource} \
        --layer -1 \
        --sets test
done

# model_name=all-in-one-cluster
# for resource in framenet propbank; do
#     for layer in 0; do
#         python verb_sense_clustering.py \
#             --model_name ${model_name} \
#             --resource ${resource} \
#             --layer ${layer}
#     done

#     python verb_sense_clustering.py \
#         --model_name ${model_name} \
#         --resource ${resource} \
#         --layer -1 \
#         --sets test
# done
