#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python run_modeling.py --batch_size 32 \
		       --model_name "navteca/roberta-base-squad2" \
		       --lr 3e-5 \
		       --n_epochs 2 \
		       --max_len 384 \
		       --n_stride 192 \
		       --use_kge T \
		       --gpus 0 1 \
		       --seed 18 \
		       --port 69420
