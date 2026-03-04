import json, os, random, argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STRUCT_BOS = "<struct>"
SEQ_BOS    = "<seq>"
AA_VOCAB   = set("ACDEFGHIKLMNPQRSTVWY")


def make_prompt(tokens_3di):
    spaced_3di = " ".join(list(tokens_3di.lower()))
    return f"{STRUCT_BOS} {spaced_3di} {SEQ_BOS}"


def setup_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})
    return tok


def load_random_base(model_name, tokenizer):
    print(f"  Loading base (RANDOM weights): {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model  = AutoModelForCausalLM.from_config(config)
    model  = model.to(torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.use_cache = False
    return model.to("cuda").eval()


def load_finetuned_random(checkpoint):
    cfg = PeftConfig.from_pretrained(checkpoint)

    tok_path = next(
        (p for p in [checkpoint, os.path.dirname(checkpoint), cfg.base_model_name_or_path]
         if os.path.exists(os.path.join(p, "tokenizer_config.json"))),
        cfg.base_model_name_or_path,
    )
    tokenizer = setup_tokenizer(tok_path)

    model = load_random_base(cfg.base_model_name_or_path, tokenizer)

    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    print(f"LoRA (random-base) merged from {checkpoint}")
    return model.eval(), tokenizer


@torch.no_grad()
def compute_recovery(model, tokenizer, records, max_length, do_sample, temperature, top_p):
    correct, total, examples = 0, 0, []

    for r in tqdm(records, desc="Recovery"):
        inputs = tokenizer(
            make_prompt(r["tokens_3di"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=r["length"] * 2 + 10,  
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

        out = model.generate(**inputs, **gen_kwargs)

        gen_text = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        pred = "".join(gen_text.split())
        pred = "".join(c for c in pred if c in AA_VOCAB)[:r["length"]]

        print(f"gen : {gen_text}")
        print(f"pred: {pred}")
        ref = r["seq"]
        print(f"ref : {ref}")

        c = sum(p == q for p, q in zip(pred, ref))
        correct += c
        total   += len(ref)
        examples.append({
            "name":     r["name"],
            "length":   len(ref),
            "pred_len": len(pred),
            "ref":      ref,
            "pred":     pred,
            "recovery": c / len(ref),
        })

    return correct / total if total else 0.0, examples


def print_results(label, recovery, examples):
    def wavg(sub):
        return (sum(e["recovery"] * e["length"] for e in sub) /
                sum(e["length"] for e in sub)) if sub else float("nan")

    short  = [e for e in examples if e["length"] <= 100]
    medium = [e for e in examples if 100 < e["length"] <= 300]
    long_  = [e for e in examples if e["length"] > 300]
    exact  = sum(1 for e in examples if e["pred_len"] == e["length"])

    print(f"\n{'='*55}\n  {label}  (n={len(examples)})\n{'='*55}")
    print(f"  Recovery (all)    : {recovery*100:6.2f}%")
    print(f"  Recovery (≤100aa) : {wavg(short)*100:6.2f}%  (n={len(short)})")
    print(f"  Recovery (101-300): {wavg(medium)*100:6.2f}%  (n={len(medium)})")
    print(f"  Recovery (>300aa) : {wavg(long_)*100:6.2f}%  (n={len(long_)})")
    print(f"  Exact length      : {exact}/{len(examples)}")
    for e in examples[:3]:
        flag = "+" if e["pred_len"] == e["length"] else f"len={e['pred_len']}"
        print(f"\n  {e['name']} rec={e['recovery']:.3f} {flag}")
        print(f"    ref : {e['ref'][:60]}")
        print(f"    pred: {e['pred'][:60]}")


def main(args):
    random.seed(42)

    records = [json.loads(l) for l in open(args.test_file) if l.strip()]
    records = [r for r in records if r.get("tokens_3di") and r.get("seq")]
    if args.n_eval > 0:
        records = random.sample(records, min(args.n_eval, len(records)))
    print(f"Evaluating {len(records)} chains")

    all_results = {}
    labels = args.labels or [f"random-base-lora ({os.path.basename(c)})"
                              for c in (args.checkpoint or [])]

    for ckpt, label in zip(args.checkpoint or [], labels):
        model, tokenizer = load_finetuned_random(ckpt)
        recovery, exs = compute_recovery(
            model, tokenizer, records, args.max_length,
            args.do_sample, args.temperature, args.top_p,
        )
        print_results(label, recovery, exs)
        all_results[label] = {
            "recovery": recovery,
            "n_eval":   len(records),
            "weights":  "random",
            "examples": exs[:10],
        }
        del model
        torch.cuda.empty_cache()

    if len(all_results) > 1:
        print(f"\n{'Model':<40} {'Recovery':>10}")
        print("-" * 52)
        for lbl, res in all_results.items():
            print(f"{lbl:<40} {res['recovery']*100:>9.2f}%")


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(all_results, open(args.out, "w"), indent=2)
    print(f"\nSaved  {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate ranndom-weight base + LoRA)"
    )
    p.add_argument("--checkpoint",   nargs="+", required=True,
                   help="Path(s) to saved LoRA checkpoint directory")
    p.add_argument("--labels",       nargs="+", default=None,
                   help="Display labels for each checkpoint (optional)")
    p.add_argument("--model_name",   default="meta-llama/Meta-Llama-3-8B",
                   help="HuggingFace model ID for the base architecture")
    p.add_argument("--test_file",    default="data/tokenized3/test.jsonl")
    p.add_argument("--out",          default="results/eval_v4_random.json")
    p.add_argument("--n_eval",       type=int, default=-1,
                   help="-1 = evaluate all records")
    p.add_argument("--max_length",   type=int, default=1024)
    p.add_argument("--do_sample",    action="store_true",
                   help="Use sampling instead of greedy decoding")
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_p",        type=float, default=0.9)
    main(p.parse_args())