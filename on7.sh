export CUDA_VISIBLE_DEVICES=2

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/UMLS_Only_NN-DTE-to-boychaboy-SNLI_roberta-base.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --fancy_concat True \
                       --gpus 2 \
                       --port 12344

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/UMLS_Only_NN-DTE-to-boychaboy-SNLI_roberta-base.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --concat_kge True \
                       --gpus 2 \
                       --port 12344

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/Mikolov_to_textattack_bert-base-uncased-snli.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --fancy_concat True \
                       --gpus 2 \
                       --port 12344

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/Mikolov_to_textattack_bert-base-uncased-snli.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --concat_kge True \
                       --gpus 2 \
                       --port 12344

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/Mikolov_to_textattack_bert-base-uncased-snli.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --fancy_concat True \
                       --random_kge True \
                       --gpus 2 \
                       --port 12344