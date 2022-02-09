export CUDA_VISIBLE_DEVICES=0,2,3

nohup python run_modeling.py --batch_size 40 \
                       --data "data/COVID-QA_cleaned_final.json" \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "NN-DTE-to-navteca-roberta-base-squad2.pkl" \
                       --lr 1e-5 \
                       --n_epochs 1 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --fancy_concat T \
                       --n_neg_records 4 \
                       --gpus 3 6 \
                       --seed 16 \
                       --port 42013 > result_biobert_kge.out &


nohup python run_modeling.py --batch_size 40 \
                       --data "data/COVID-QA_cleaned_final.json" \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "Mikolov++_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 2e-5 \
                       --n_epochs 1 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --random_kge T\
                       --fancy_concat T \
                       --n_neg_records 4 \
                       --gpus 3 4 \
                       --seed 16 \
                       --port 42020 > result_scibert_kge.out &


nohup python run_modeling.py --batch_size 40 \
                       --data "data/COVID-QA_cleaned_final.json" \
                       --model_name "navteca/roberta-base-squad2" \
                       --dte_lookup_table_fp "NN-DTE-to-navteca-roberta-base-squad2.pkl" \
                       --lr 2e-5 \
                       --n_epochs 1 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --random_kge T \
                       --fancy_concat T \
                       --n_neg_records 4 \
                       --gpus 0 2 3 \
                       --seed 16 \
                       --port 42068 > result_roberta.out &


python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
                       --dte_lookup_table_fp "NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl" \
                       --lr 2e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 4 \
                       --gpus 1 2 \
                       --seed 16 \
                       --port 42090


export CUDA_VISIBLE_DEVICES=3,6
nohup python run_modeling.py --batch_size 40 \
                       --model_name "ktrapeznikov/scibert_scivocab_uncased_squad_v2" \
                       --dte_lookup_table_fp "NN-DTE-to-ktrapeznikov-scibert_scivocab_uncased_squad_v2.pkl" \
                       --lr 2e-5 \
                       --n_epochs 1 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 4 \
                       --concat_kge True \
                       --use_kge True \
                       --use_dict False \
                       --gpus 3 6 \
                       --seed 16 \
                       --port 42056 > results_all.out &


                       

python pykg2vec_tune.py -mn TransE -ds UMLS_KG_MT -dsp /home/Train_KGE/UMLS_KG_MT-original -device cuda
python pykg2vec_train.py -mn TransE -ds UMLS_KG_MT -dsp /home/Train_KGE/UMLS_KG_MT-original -device cuda
python pykg2vec_train.py -mn TransE -ds UMLS_KG_MT -dsp /home/Train_KGE/UMLS_KG_MT-original -device cuda -b 2769 -l 10 -k 170 -l1 True -lr 0.030143630391557222 -mg 0.2616645450619097 -opt sgd

python pykg2vec_train.py -mn DistMult -ds UMLS_KG_MT -dsp /home/Train_KGE/UMLS_KG_MT -device cuda -b 3037 -l 10 -k 135 -lr 0.024743441143928905 -lmda 1.384683308307553e-05 -opt rms


'batch_size': 451, 'epochs': 10, 'hidden_size': 79, 'l1_flag': True, 'learning_rate': 0.0023120564129837724, 'margin': 0.23949684100171093, 'optimizer': 'rms'