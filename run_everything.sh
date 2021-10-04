#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

######BERT Experiments######

#Mikolov Replace BERT
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --dte_lookup_table_fp "Mikolov_to_phiyodr_bert-base-finetuned-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Replace BERT
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --dte_lookup_table_fp "Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

######RoBERTa Experiments######

#Random Concat with RoBERTa
python run_modeling.py --batch_size 40 \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "Mikolov_to_navteca_roberta-base-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --random_kge T \
                       --n_neg_records 10 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069

#Mikolov Replace RoBERTa
python run_modeling.py --batch_size 40 \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "Mikolov_to_navteca_roberta-base-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Concat RoBERTa
python run_modeling.py --batch_size 40 \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "Mikolov_to_navteca_roberta-base-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Replace RoBERTa
python run_modeling.py --batch_size 40 \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "Mikolov++_to_navteca_roberta-base-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42066

######BioBERT Experiments######

#Vanilla BioBERT FT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 10 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069

#Random Concat with BioBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --random_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Replace BioBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Concat BioBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Replace BioBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov++_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Concat BioBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "Mikolov++_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

######SciBERT Experiments######

#Vanilla SciBERT FT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 10 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069

#Random Concat with SciBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --random_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Replace SciBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Concat SciBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Replace SciBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov++_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Concat SciBERT
python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov++_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

######PubMedBERT Experiments######

#Vanilla PubMedBERT FT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 10 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069

#Random Concat with PubMedBERT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --random_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Replace PubMedBERT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov Concat PubMedBERT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Replace PubMedBERT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov++_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069

#Mikolov++ Concat PubMedBERT
python run_modeling.py --batch_size 40 \
                       --model_name "franklu/pubmed_bert_squadv2" \
                       --dte_lookup_table_fp "Mikolov++_to_franklu_pubmed_bert_squadv2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 10 \
                       --gpus 0 7 \
                       --seed 16 \
                       --port 42069



