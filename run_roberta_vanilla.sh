#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
python run_modeling.py --batch_size 24 \
		       --model_name "deepset/roberta-base-squad2" \
		       --lr 3e-5 \
		       --n_epochs 2 \
		       --max_len 384 \
		       --n_stride 196 \
		       --warmup_proportion 0.1 \
		       --use_kge F \
		       --gpus 0 1 2 \
		       --seed 18 \
		       --port 42069