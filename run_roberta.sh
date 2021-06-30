#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run_modeling.py --batch_size 32 \
		       --model_name "navteca/roberta-base-squad2" \
		       --lr 1e-4 \
		       --n_epochs 2 \
		       --use_kge T
