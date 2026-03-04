# 1
conda env create -f environment.yml

conda activate pifold_env

./download_data.sh

# 2
JSON to PDB to 3di tokens

python filteredjsonto3di.py --data_dir data/cath --out_dir data/tokenized

JSON to Calpha tokens

python filteredjsontocalpha.py --data_dir data/cath --out_dir data/tokenized

# 3
finetuning 

python finetuning/INSTRUCT_3di.py --data_dir data/tokenized --out_dir results/

# 4
evals

python evals/eval_INSTRCUT_3di.py --checkpoint PATH/TO/CHECKPOINT --test_file data/tokenized/test.jsonl


