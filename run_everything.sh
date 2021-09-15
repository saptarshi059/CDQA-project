#!/bin/bash

#Training optimal distmult for MT
python ../Train_KGE/pykg2vec/scripts/pykg2vec_train.py -mn DistMult -ds UMLS_KG_MT -dsp /home/Train_KGE/UMLS_KG_MT -device cuda -b 3213 -l 10 -k 8 -lr 0.006543608492240101 -lmda 6.082386885464442e-05 -opt rms

#Homogenize using Mikolov Strategy
python Mikolov\(E-BERT\)\ approach/Translation_Approach.py --UMLS_Path ../Train_KGE/UMLS_KG_MT --BERT_Variant phiyodr/bert-base-finetuned-squad2

export CUDA_VISIBLE_DEVICES=3,5
nohup python run_modeling.py --batch_size 40 \
                       --model_name 'phiyodr/bert-base-finetuned-squad2' \
                       --dte_lookup_table_fp 'Mikolov_to_phiyodr_bert-base-finetuned-squad2.pkl' \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 5 \
                       --gpus 3 5\
                       --seed 16 \
                       --port 42069 > "bert_MIK_MT.out" &

rm Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl

#### One done
python Mikolov\(E-BERT\)\ approach/Translation_Approach.py --UMLS_Path ../Train_KGE/UMLS_KG_SN --BERT_Variant phiyodr/bert-base-finetuned-squad2

export CUDA_VISIBLE_DEVICES=3,5
nohup python run_modeling.py --batch_size 40 \
                       --model_name 'phiyodr/bert-base-finetuned-squad2' \
                       --dte_lookup_table_fp 'Mikolov_to_phiyodr_bert-base-finetuned-squad2.pkl' \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 5 \
                       --gpus 3 5\
                       --seed 16 \
                       --port 42069 > "bert_MIK_SN.out" &

rm Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl

####Next

python Homogenization_Programs/Mikolov++/Mik4KG.py --UMLS_Path ../Train_KGE/UMLS_KG_SN --BERT_Variant phiyodr/bert-base-finetuned-squad2 --KGE_Variant distmult --THROUGH True --TRIPLES False --HS False --PO True

export CUDA_VISIBLE_DEVICES=3,5
nohup python run_modeling.py --batch_size 40 \
                       --model_name 'phiyodr/bert-base-finetuned-squad2' \
                       --dte_lookup_table_fp 'Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl' \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 5 \
                       --gpus 3 5\
                       --seed 16 \
                       --port 42069 > "bert_MIK++_SN.out" &

rm Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl

####Next

python Homogenization_Programs/Mikolov++/Mik4KG.py --UMLS_Path ../Train_KGE/UMLS_KG_MT+SN --BERT_Variant phiyodr/bert-base-finetuned-squad2 --KGE_Variant distmult --THROUGH True --TRIPLES False --HS False --PO True

export CUDA_VISIBLE_DEVICES=3,5
nohup python run_modeling.py --batch_size 40 \
                       --model_name 'phiyodr/bert-base-finetuned-squad2' \
                       --dte_lookup_table_fp 'Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl' \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 5 \
                       --gpus 3 5\
                       --seed 16 \
                       --port 42069 > "bert_MIK++_MT+SN.out" &

rm Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl