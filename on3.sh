export CUDA_VISIBLE_DEVICES=2

#RoBERTa tests


python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/NN-DTE-to-boychaboy-SNLI_roberta-base.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --concat_kge True \
                       --gpus 2 \
                       --port 11312

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/Definition_Only_NN-DTE-to-boychaboy-SNLI_roberta-base.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --fancy_concat True \
                       --gpus 0 \
                       --port 11344

python run_pubmed_qa.py --model_name "boychaboy/SNLI_roberta-base" \
                       --dte_lookup_table_fp /home/CDQA-project/pretrained\ stuff/for_pubmedqa/roberta/Definition_Only_NN-DTE-to-boychaboy-SNLI_roberta-base.pkl \
                       --data /home/CDQA-project/data/ \
                       --n_epochs 5 \
                       --use_kge True \
                       --concat_kge True \
                       --gpus 0 \
                       --port 11344