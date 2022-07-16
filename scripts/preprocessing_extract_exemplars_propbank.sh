#!/bin/sh

python ../prprocessing/extract_exemplars_propbank.py \
    --input_frame_path ../data/raw/ONTONOTES/metadata/frames \
    --input_anno_path ../data/raw/ONTONOTES/annotations \
    --output_path ../data/preprocessing/propbank
