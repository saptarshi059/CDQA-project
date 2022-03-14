for (( epoch=0; epoch <= 4; epoch++ ))
do
    python apply_pubmed_qa.py --data /home/CDQA-project/data/test_set.json --model_id "$1" --epoch "$epoch"
done