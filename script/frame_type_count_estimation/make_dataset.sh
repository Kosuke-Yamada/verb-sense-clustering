#!/bin/sh

source_dir=../../source/frame_type_count_estimation
data_dir=../../data/frame_type_count_estimation
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
