#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python run_modeling.py --batch_size 32 \
		       --model_name "navteca/roberta-base-squad2" \
		       --lr 3e-5 \
		       --n_epochs 2 \
		       --n_stride 192 \
		       --use_kge T