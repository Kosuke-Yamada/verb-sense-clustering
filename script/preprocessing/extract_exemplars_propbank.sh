#!/bin/sh

python ../prprocessing/extract_exemplars_propbank.py \
    --input_frame_path ../data/raw/ontonotes/metadata/frames \
    --input_anno_path ../data/raw/ontonotes/annotations \
    --output_path ../data/preprocessing/propbank
