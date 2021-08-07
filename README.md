# CDQA-project #

Programs to execute in order:
KG_Constructor.ipynb \
Train_KGE.ipynb \
Entity_Expansion.ipynb \
KGE_Homogenization.ipynb


# Commands #
To change which model is use, please make sure to change the ```--model_name``` and ```--dte_lookup_table_fp``` args.
All of the various KGE parms should be able to be used together.

**Note:** the ```--gpus``` expects consecutive GPU ID's that  **always** start with 0. These ID's are considered wrt the
visible devices, as defined by a ```export CUDA_VISIBLE_DEVICES=...``` command. Even if you want to use, say, GPU 0 and 
1, I believe it is preferred to explicitly do ```export CUDA_VISIBLE_DEVICES=0,1```. So, to use GPUs not starting w/ the 
0 ID, such as GPUs 2 and 3, use the command ```export CUDA_VISIBLE_DEVICES=2,3``` and ```run_modeling.py``` arg of 
```gpus 0 1```. An example in a ```.sh``` script is given below.

```commandline
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python run_modeling.py --gpus 0 1 
```


**Note:** to have the script run in the background and output to a file, use a ```bash``` script akin to the below:
```commandline
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
nohup python run_modeling.py --gpus 0 1 > "run_model.out" & 
```

The output will be written to the file whenever the pipe is full (I think? lol). A convenient way to frequently check
the file is ```watch -n 5 "cat run_model.out | tail -n 30"```.

**Note: If you are training 2+ models at the same time, make sure that no two training scripts are given the same 
```--port``` arg (doesn't apply to baseline models).**


### BERT Baseline ###
```commandline
python covid_qa_baseline.py --model_name "phiyodr/bert-base-finetuned-squad2" \
                            --dte_lookup_table_fp "DTE_to_phiyodr_bert-base-finetuned-squad2.pkl" \
                            --max_len 384 \
                            --n_stride 196
```

### Vanilla BERT Fine-tuning ###
```commandline
export CUDA_VISIBLE_DEVICES=0,1
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --dte_lookup_table_fp "DTE_to_phiyodr_bert-base-finetuned-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --n_neg_records 2 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069
```

### BERT+KGE-Replace Fine-tuning ###
```commandline
export CUDA_VISIBLE_DEVICES=0,1
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --dte_lookup_table_fp "DTE_to_phiyodr_bert-base-finetuned-squad2.pkl" \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --n_neg_records 3 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069
```


### BERT+Random-KGE-Replace Fine-tuning ###
```commandline
export CUDA_VISIBLE_DEVICES=0,1
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --lr 3e-5 \
                       --n_epochs 2 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --random_kge T \
                       --n_neg_records 2 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069
```

### BERT+KGE-Concat Fine-tuning ###
```commandline
export CUDA_VISIBLE_DEVICES=0,1
python run_modeling.py --batch_size 40 \
                       --model_name "phiyodr/bert-base-finetuned-squad2" \
                       --lr 3e-5 \
                       --n_epochs 3 \
                       --max_len 384 \
                       --n_stride 196 \
                       --warmup_proportion 0.1 \
                       --use_kge T \
                       --concat_kge T \
                       --n_neg_records 5 \
                       --gpus 0 1 \
                       --seed 16 \
                       --port 42069
```
