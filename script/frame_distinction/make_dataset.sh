#!/bin/sh

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction
input_dir=../../data/preprocessing

resource=framenet
#resource=propbank

python ${source_dir}/make_dataset.py \
    --input_path ${input_dir} \
    --output_path ${data_dir}/dataset \
    --resource ${resource} \
    --min_lu 2 \
    --max_lu 10 \
    --min_text 20 \
    --max_text 100
