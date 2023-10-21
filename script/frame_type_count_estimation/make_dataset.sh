#!/bin/sh

resource=framenet
#resource=propbank

python ../experiment_frame_number_estimation/make_dataset.py \
    --input_path ../data/preprocessing \
    --output_path ../data/experiment_frame_number_estimation/dataset \
    --resource ${resource} \
    --min_lu 2 \
    --max_lu 10 \
    --min_text 20 \
    --max_text 100
