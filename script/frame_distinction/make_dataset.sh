#!/bin/bash

source_dir=../../source/frame_distinction
data_dir=../../data/frame_distinction
input_dir=../../data/preprocessing

# resources=(framenet propbank)
resources=(framenet)
# resources=(propbank)

for resource in ${resources[@]}; do
    d1=${resource}
    python ${source_dir}/make_dataset.py \
        --input_file ${input_dir}/${d1}/exemplars.jsonl \
        --output_dir ${data_dir}/dataset/${d1} \
        --min_lu 2 \
        --max_lu 10 \
        --min_text 20 \
        --max_text 100
done
