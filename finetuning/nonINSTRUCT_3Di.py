import json, os, random, argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

STRUCT_BOS = "<struct>"
SEQ_BOS    = "<seq>"

torch.set_float32_matmul_precision("medium") 

def make_prompt(tokens_3di, aa_seq=""):
    spaced_3di = " ".join(list(tokens_3di.lower()))
    if aa_seq:
        spaced_aa = " ".join(list(aa_seq.upper()))
        return f"{STRUCT_BOS} {spaced_3di} {SEQ_BOS} {spaced_aa}"
    return f"{STRUCT_BOS} {spaced_3di} {SEQ_BOS}"


def setup_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})

    test_ids = tok.encode(" A C D E", add_special_tokens=False)
    print(f"  Tokenizer check ' A C D E'  {len(test_ids)} token")
    return tok


class InverseFoldingDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, n_samples=-1):
        records = [json.loads(l) for l in open(path) if l.strip()]
        records = [r for r in records if r.get("tokens_3di") and r.get("seq")]
        if n_samples > 0:
            random.shuffle(records)
            records = records[:n_samples]
        self.records, self.tokenizer, self.max_length = records, tokenizer, max_length
        print(f"  {len(records)} examples from {os.path.basename(path)}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        enc = self.tokenizer(
            make_prompt(r["tokens_3di"], r["seq"]),
            truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None, add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        labels    = list(input_ids)

        seq_bos_id = self.tokenizer.convert_tokens_to_ids(SEQ_BOS)
        if seq_bos_id in input_ids:
            n_prompt = input_ids.index(seq_bos_id) + 1  
        else:
            n_prompt = len(self.tokenizer.encode(
                make_prompt(r["tokens_3di"]), add_special_tokens=True
            ))

        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100

        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            labels    = labels    + [self.tokenizer.eos_token_id]

        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         labels,
        }


def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Tokenizer: {args.model_name}")
    tokenizer = setup_tokenizer(args.model_name)

    print(f"Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=args.lora_dropout, #0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Datasets:")
    train_ds = InverseFoldingDataset(
        os.path.join(args.data_dir, "train.jsonl"), tokenizer, args.max_length, args.n_train,
    )
    val_ds = InverseFoldingDataset(
        os.path.join(args.data_dir, "validation.jsonl"), tokenizer, args.max_length,
    )

    # snan check
    s = train_ds[0]
    n_label = sum(1 for label in s["labels"] if label != -100)
    assert n_label > 0, "All labels masked — prompt masking bug"
    print(f"  sanity: input_len={len(s['input_ids'])}  label_tokens={n_label}")

    n_gpus    = max(torch.cuda.device_count(), 1)
    eff_batch = args.batch_size * args.grad_accum * n_gpus
    print(f"  GPUs={n_gpus}  eff_batch={eff_batch}  steps/epoch≈{max(1, len(train_ds)//eff_batch)}")

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False, 
        max_grad_norm=1.0,
        weight_decay=0.01,
        # dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model,
            padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
        ),
    )
    trainer.train()

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved  {args.out_dir}")

    json.dump({
        "model": args.model_name, "n_train": len(train_ds),
        "lora_r": args.lora_r, "lr": args.lr, "epochs": args.epochs,
        "prompt_format": "spaced_residues",
    }, open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--data_dir",   default="data/tokenized3")
    p.add_argument("--out_dir",    default="results/llm_3di_v2")
    p.add_argument("--n_train",    type=int,   default=-1)
    p.add_argument("--max_length", type=int,   default=512)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--grad_accum", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lora_r",     type=int,   default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    main(p.parse_args())