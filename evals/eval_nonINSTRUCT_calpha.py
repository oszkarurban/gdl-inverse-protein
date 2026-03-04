import json, os, math, random, argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STRUCT_BOS = "<struct>"
SEQ_BOS    = "<seq>"
AA_VOCAB   = set("ACDEFGHIKLMNPQRSTVWY")


def format_coords(ca_coords, decimal_places=1):
    arr = np.array(ca_coords, dtype=np.float32)
    arr = arr - arr.mean(axis=0)
    fmt = f".{decimal_places}f"
    return " ".join(f"{v:{fmt}}" for xyz in arr for v in xyz)


def make_prompt(ca_coords, decimal_places=1):
    return f"{STRUCT_BOS} {format_coords(ca_coords, decimal_places)} {SEQ_BOS}"


def make_full_sequence(ca_coords, seq, decimal_places=1):
    spaced_seq = " ".join(list(seq.upper()))
    return f"{make_prompt(ca_coords, decimal_places)} {spaced_seq}"


def setup_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})
    test_ids = tok.encode(" 12.3 -4.5 8.1", add_special_tokens=False)
    print(f"  Tokenizer check ' 12.3 -4.5 8.1'  {len(test_ids)} tokens")
    return tok


def load_base(model_name, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,      
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",               
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    return model.eval()


def load_finetuned(checkpoint):
    cfg = PeftConfig.from_pretrained(checkpoint)
    tok_path = next(
        (p for p in [checkpoint, os.path.dirname(checkpoint), cfg.base_model_name_or_path]
         if os.path.exists(os.path.join(p, "tokenizer_config.json"))),
        cfg.base_model_name_or_path,
    )
    tokenizer = setup_tokenizer(tok_path)
    model     = load_base(cfg.base_model_name_or_path, tokenizer)
    model     = PeftModel.from_pretrained(
        model, checkpoint, device_map="auto",
    ).merge_and_unload()
    print(f"LoRA merged from {checkpoint}")
    return model.eval(), tokenizer


def load_zeroshot(model_name):
    tokenizer = setup_tokenizer(model_name)
    model     = load_base(model_name, tokenizer)
    print(f"Zero-shot {model_name}")
    return model, tokenizer


@torch.no_grad()
def compute_perplexity_for_record(model, tokenizer, r, max_length, decimal_places):
    prompt   = make_prompt(r["ca_coords"], decimal_places)
    full_str = make_full_sequence(r["ca_coords"], r["seq"], decimal_places)

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).input_ids
    full_ids = tokenizer(
        full_str, return_tensors="pt", truncation=True, max_length=max_length,
    ).input_ids

    prompt_len = prompt_ids.shape[1]
    n_residues = full_ids.shape[1] - prompt_len
    if n_residues <= 0:
        return float("nan"), 0

    first_device = next(model.parameters()).device
    full_ids = full_ids.to(first_device)
    labels   = full_ids.clone()
    labels[0, :prompt_len] = -100

    outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss.item(), n_residues


@torch.no_grad()
def compute_recovery(model, tokenizer, records, max_length, decimal_places):
    correct, total, examples = 0, 0, []
    first_device = next(model.parameters()).device

    for r in tqdm(records, desc="Eval"):

        inputs = tokenizer(
            make_prompt(r["ca_coords"], decimal_places),
            return_tensors="pt", truncation=True,
            max_length=max_length, add_special_tokens=True,
        ).to(first_device)

        out = model.generate(
            **inputs,
            max_new_tokens=r["length"] + 10,
            do_sample=False, temperature=None, top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen  = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = "".join(gen.split())
        pred = "".join(c for c in pred if c in AA_VOCAB)[:r["length"]]
        ref  = r["seq"]

        print(f"gen : {gen!r}")
        print(f"pred: {pred[:60]}")
        print(f"ref : {ref[:60]}")

        c = sum(p == q for p, q in zip(pred, ref))
        correct += c; total += len(ref)

        nll, n_res = compute_perplexity_for_record(
            model, tokenizer, r, max_length, decimal_places,
        )
        ppl = math.exp(nll) if not math.isnan(nll) else float("nan")
        print(f"ppl : {ppl:.3f}  (nll={nll:.4f}, residues={n_res})")

        examples.append({
            "name": r["name"], "length": len(ref), "pred_len": len(pred),
            "ref": ref, "pred": pred, "recovery": c / len(ref),
            "nll": nll, "perplexity": ppl, "n_residues": n_res,
        })

    overall_recovery = correct / total if total else 0.0
    valid   = [e for e in examples if not math.isnan(e["nll"])]
    avg_ppl = (math.exp(sum(e["nll"]*e["n_residues"] for e in valid) /
                        sum(e["n_residues"] for e in valid))
               if valid else float("nan"))
    return overall_recovery, avg_ppl, examples


def _wavg_rec(sub):
    return (sum(e["recovery"]*e["length"] for e in sub) / sum(e["length"] for e in sub)
            if sub else float("nan"))

def _wavg_ppl(sub):
    valid = [e for e in sub if not math.isnan(e["nll"])]
    if not valid: return float("nan")
    return math.exp(sum(e["nll"]*e["n_residues"] for e in valid) /
                    sum(e["n_residues"] for e in valid))


def print_results(label, recovery, avg_ppl, examples):
    short  = [e for e in examples if e["length"] <= 100]
    medium = [e for e in examples if 100 < e["length"] <= 300]
    long_  = [e for e in examples if e["length"] > 300]
    exact  = sum(1 for e in examples if e["pred_len"] == e["length"])

    print(f"\n{'='*60}\n  {label}  (n={len(examples)})\n{'='*60}")
    print(f"  Recovery (all)     : {recovery*100:6.2f}%")
    print(f"  Recovery (≤100aa)  : {_wavg_rec(short)*100:6.2f}%  (n={len(short)})")
    print(f"  Recovery (101-300) : {_wavg_rec(medium)*100:6.2f}%  (n={len(medium)})")
    print(f"  Recovery (>300aa)  : {_wavg_rec(long_)*100:6.2f}%  (n={len(long_)})")
    print(f"  Exact length       : {exact}/{len(examples)}")
    print()
    print(f"  Perplexity (all)   : {avg_ppl:8.3f}  [residue-level]")
    print(f"  Perplexity (≤100)  : {_wavg_ppl(short):8.3f}  (n={len(short)})")
    print(f"  Perplexity (101-300): {_wavg_ppl(medium):8.3f}  (n={len(medium)})")
    print(f"  Perplexity (>300)  : {_wavg_ppl(long_):8.3f}  (n={len(long_)})")

    for e in examples[:3]:
        flag = "+" if e["pred_len"] == e["length"] else f"len={e['pred_len']}"
        print(f"\n  {e['name']}  rec={e['recovery']:.3f}  ppl={e['perplexity']:.2f}  {flag}")
        print(f"    ref : {e['ref'][:60]}")
        print(f"    pred: {e['pred'][:60]}")


def main(args):
    random.seed(42)
    records = [json.loads(l) for l in open(args.test_file) if l.strip()]
    records = [r for r in records if r.get("ca_coords") and r.get("seq")]
    if args.n_eval > 0:
        records = random.sample(records, min(args.n_eval, len(records)))
    print(f"Evaluating {len(records)} chains")

    all_results = {}

    def run(model, tokenizer, label):
        recovery, avg_ppl, exs = compute_recovery(
            model, tokenizer, records, args.max_length, args.decimal_places,
        )
        print_results(label, recovery, avg_ppl, exs)
        all_results[label] = {
            "recovery": recovery, "perplexity": avg_ppl,
            "n_eval": len(records), "examples": exs[:10],
        }
        del model; torch.cuda.empty_cache()

    if args.zero_shot:
        m, t = load_zeroshot(args.model_name)
        run(m, t, f"zero-shot ({args.model_name.split('/')[-1]})")

    labels = args.labels or args.checkpoint or []
    for ckpt, label in zip(args.checkpoint or [], labels):
        m, t = load_finetuned(ckpt)
        run(m, t, label)

    if len(all_results) > 1:
        print(f"\n{'Model':<35} {'Recovery':>10} {'Perplexity':>12}")
        print("-" * 62)
        for lbl, res in all_results.items():
            print(f"  {lbl:<33} {res['recovery']*100:>9.2f}%  {res['perplexity']:>11.3f}")


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(all_results, open(args.out, "w"), indent=2)
    print(f"\nSaved  {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     nargs="+", default=None)
    p.add_argument("--labels",         nargs="+", default=None)
    p.add_argument("--zero_shot",      action="store_true")
    p.add_argument("--model_name",     default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--test_file",      default="data/tokenized_ca/test.jsonl")
    p.add_argument("--out",            default="results/eval_NONinstruct_calpha.json")
    p.add_argument("--n_eval",         type=int, default=-1)
    p.add_argument("--max_length",     type=int, default=3072)
    p.add_argument("--decimal_places", type=int, default=1)
    args = p.parse_args()
    if not args.zero_shot and not args.checkpoint:
        p.error("Provide --checkpoint and/or --zero_shot")
    main(args)