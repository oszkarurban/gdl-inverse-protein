# Assessing the Transferability of Text LLMs for Protein Inverse Folding via Structural Tokenization

This work investigates the transferability of linguistic priors from large-scale
text models (Llama-3-8B) to the task of inverse protein folding. By discretizing
3D backbone structures into 1D tokens via Foldseek’s 3Di alphabet and C-α
coordinate-string representations, we evaluate the model’s ability to design amino
acid sequences. Our results indicate that while few-shot recovery performance is
minimal, LoRA fine-tuning achieves improved sequence recovery, demonstrating
that linguistic priors from large-scale text pre-training transfer to the protein design
domain. Although overall performance remains below state-of-the-art geometric
approaches, our ablation against randomly initialized model weights confirms that
large-scale pretraining, and the inductive biases of large transformer architectures,
provides a necessary foundation for effective adaptation to downstream tasks,
without modifications to the transformer architecture

---

# Environment setup
conda env create -f environment.yml

conda activate pifold_env

./download_data.sh

# Dataset
JSON to PDB to 3di tokens

python filteredjsonto3di.py --data_dir data/cath --out_dir data/tokenized

JSON to Calpha tokens

python filteredjsontocalpha.py --data_dir data/cath --out_dir data/tokenized

---

# Training
python finetuning/INSTRUCT_3di.py --data_dir data/tokenized --out_dir results/

# Evaluation
evals

python evals/eval_INSTRCUT_3di.py --checkpoint PATH/TO/CHECKPOINT --test_file data/tokenized/test.jsonl


