import json, os, math, random, argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STRUCT_BOS = "<struct>"
SEQ_BOS    = "<seq>"
AA_VOCAB   = set("ACDEFGHIKLMNPQRSTVWY")

SYSTEM_3DI = (
    "Your task is to predict an amino acid sequence from a protein structure "
    "encoded as a sequence of 3Di structural alphabet tokens produced by Foldseek.\n\n"
    "INPUT FORMAT:\n"
    "You receive a sequence of 3Di tokens, where each token is a single lower-case letter "
    "from the 3Di structural alphabet (20 possible letters: acdefghiklmnpqrstvwy). "
    "Each token encodes the local geometric environment of one residue, "
    "specifically the backbone and side-chain conformation relative to its neighbours. "
    "Tokens are separated by spaces, one token per residue.\n\n"
    "OUTPUT FORMAT:\n"
    "Respond with the predicted amino acid sequence. "
    "Each amino acid is represented as a single upper-case one-letter code "
    "(standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY). "
    "The output sequence must have exactly the same number of residues as the input 3Di sequence. "
    "Output only the <seq> token followed by the space-separated amino acid sequence, nothing else."
)

SYSTEM_CALPHA = (
    "Your task is to predict an amino acid sequence from a protein structure "
    "encoded as Ca (alpha-carbon) coordinates.\n\n"
    "INPUT FORMAT:\n"
    "You receive a sequence of Ca coordinates, one triplet per residue, "
    "formatted as space-separated decimal numbers: x y z x y z ... "
    "Coordinates are centroid-normalised (mean-subtracted across the chain) "
    "and rounded to 1 decimal place in Angstroms.\n\n"
    "OUTPUT FORMAT:\n"
    "Respond with the predicted amino acid sequence. "
    "Each amino acid is represented as a single upper-case one-letter code "
    "(standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY). "
    "The output sequence must have exactly the same number of residues as the input coordinate triplets. "
    "Output only the <seq> token followed by the space-separated amino acid sequence, nothing else."
)

def spaced_3di(tokens_3di: str) -> str:
    return " ".join(list(tokens_3di.lower()))

def spaced_aa(seq: str) -> str:
    return " ".join(list(seq.upper()))

def format_coords(ca_coords: list, decimal_places: int = 1) -> str:
    arr = np.array(ca_coords, dtype=np.float32)
    arr = arr - arr.mean(axis=0)
    fmt = f".{decimal_places}f"
    return " ".join(f"{v:{fmt}}" for xyz in arr for v in xyz)

def setup_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})
    return tok

def load_model(model_name: str, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    print(f"  Loaded {model_name}")
    return model.to("cuda").eval()

def build_prompt_base_3di(record, few_shot_examples):
    parts = []
    for ex in few_shot_examples:
        parts.append(f"{STRUCT_BOS} {spaced_3di(ex['tokens_3di'])} {SEQ_BOS} {spaced_aa(ex['seq'])}")
    parts.append(f"{STRUCT_BOS} {spaced_3di(record['tokens_3di'])} {SEQ_BOS}")
    return "\n".join(parts)

def build_prompt_base_calpha(record, few_shot_examples, decimal_places):
    parts = []
    for ex in few_shot_examples:
        parts.append(f"{STRUCT_BOS} {format_coords(ex['ca_coords'], decimal_places)} {SEQ_BOS} {spaced_aa(ex['seq'])}")
    parts.append(f"{STRUCT_BOS} {format_coords(record['ca_coords'], decimal_places)} {SEQ_BOS}")
    return "\n".join(parts)

def build_prompt_instruct_3di(record, few_shot_examples, tokenizer):
    messages = [{"role": "system", "content": SYSTEM_3DI}]
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": f"{STRUCT_BOS} {spaced_3di(ex['tokens_3di'])}"})
        messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(ex['seq'])}"})
    messages.append({"role": "user", "content": f"{STRUCT_BOS} {spaced_3di(record['tokens_3di'])}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_prompt_instruct_calpha(record, few_shot_examples, tokenizer, decimal_places):
    messages = [{"role": "system", "content": SYSTEM_CALPHA}]
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": f"{STRUCT_BOS} {format_coords(ex['ca_coords'], decimal_places)}"})
        messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(ex['seq'])}"})
    messages.append({"role": "user", "content": f"{STRUCT_BOS} {format_coords(record['ca_coords'], decimal_places)}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_full_base_3di(record, few_shot_examples):
    """Inference prompt + ground-truth sequence appended."""
    parts = []
    for ex in few_shot_examples:
        parts.append(f"{STRUCT_BOS} {spaced_3di(ex['tokens_3di'])} {SEQ_BOS} {spaced_aa(ex['seq'])}")
    parts.append(f"{STRUCT_BOS} {spaced_3di(record['tokens_3di'])} {SEQ_BOS} {spaced_aa(record['seq'])}")
    return "\n".join(parts)

def build_full_base_calpha(record, few_shot_examples, decimal_places):
    parts = []
    for ex in few_shot_examples:
        parts.append(f"{STRUCT_BOS} {format_coords(ex['ca_coords'], decimal_places)} {SEQ_BOS} {spaced_aa(ex['seq'])}")
    parts.append(f"{STRUCT_BOS} {format_coords(record['ca_coords'], decimal_places)} {SEQ_BOS} {spaced_aa(record['seq'])}")
    return "\n".join(parts)

def build_full_instruct_3di(record, few_shot_examples, tokenizer):
    messages = [{"role": "system", "content": SYSTEM_3DI}]
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": f"{STRUCT_BOS} {spaced_3di(ex['tokens_3di'])}"})
        messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(ex['seq'])}"})
    messages.append({"role": "user",      "content": f"{STRUCT_BOS} {spaced_3di(record['tokens_3di'])}"})
    messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(record['seq'])}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def build_full_instruct_calpha(record, few_shot_examples, tokenizer, decimal_places):
    messages = [{"role": "system", "content": SYSTEM_CALPHA}]
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": f"{STRUCT_BOS} {format_coords(ex['ca_coords'], decimal_places)}"})
        messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(ex['seq'])}"})
    messages.append({"role": "user",      "content": f"{STRUCT_BOS} {format_coords(record['ca_coords'], decimal_places)}"})
    messages.append({"role": "assistant", "content": f"{SEQ_BOS} {spaced_aa(record['seq'])}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def get_prompt_len_base(inference_ids):
    return len(inference_ids)


def get_prompt_len_instruct(inference_ids, full_ids, tokenizer):
    seq_bos_id = tokenizer.convert_tokens_to_ids(SEQ_BOS)
    for i in range(len(inference_ids), len(full_ids)):
        if full_ids[i] == seq_bos_id:
            return i + 1   
    raise ValueError(
        f"<seq> id={seq_bos_id} not found after position {len(inference_ids)}. "
        f"Tokens: {tokenizer.convert_ids_to_tokens(full_ids[len(inference_ids):len(inference_ids)+10])}"
    )

def sample_few_shot_pool(train_file, modality, k):
    if k == 0:
        return []
    key = "tokens_3di" if modality.startswith("3di") else "ca_coords"
    records = [json.loads(l) for l in open(train_file) if l.strip()]
    records = [r for r in records if r.get(key) and r.get("seq")]
    if len(records) < k:
        raise ValueError(f"Not enough training examples (found {len(records)}, need {k}).")
    pool = random.sample(records, k)
    print(f"  Sampled {k} few-shot example(s) from {train_file}")
    for p in pool:
        print(f"    - {p['name']}  len={len(p['seq'])}")
    return pool

@torch.no_grad()
def compute_perplexity_for_record(model, tokenizer, r, few_shot_pool,
                                   modality, max_length, decimal_places):
    is_instruct = modality.endswith("-inst")
    is_calpha   = modality.startswith("calpha")
    if is_instruct and is_calpha:
        inference_str = build_prompt_instruct_calpha(r, few_shot_pool, tokenizer, decimal_places)
        full_str      = build_full_instruct_calpha(r, few_shot_pool, tokenizer, decimal_places)
    elif is_instruct:
        inference_str = build_prompt_instruct_3di(r, few_shot_pool, tokenizer)
        full_str      = build_full_instruct_3di(r, few_shot_pool, tokenizer)
    elif is_calpha:
        inference_str = build_prompt_base_calpha(r, few_shot_pool, decimal_places)
        full_str      = build_full_base_calpha(r, few_shot_pool, decimal_places)
    else:
        inference_str = build_prompt_base_3di(r, few_shot_pool)
        full_str      = build_full_base_3di(r, few_shot_pool)

    add_special = not is_instruct

    inference_ids = tokenizer(
        inference_str, return_tensors="pt",
        truncation=True, max_length=max_length,
        add_special_tokens=add_special,
    ).input_ids
    full_ids = tokenizer(
        full_str, return_tensors="pt",
        truncation=True, max_length=max_length,
        add_special_tokens=add_special,
    ).input_ids

    if is_instruct:
        prompt_len = get_prompt_len_instruct(
            inference_ids[0].tolist(), full_ids[0].tolist(), tokenizer,
        )
    else:
        prompt_len = get_prompt_len_base(inference_ids[0].tolist())

    n_residues = full_ids.shape[1] - prompt_len
    if n_residues <= 0:
        return float("nan"), 0

    full_ids = full_ids.to(model.device)
    labels   = full_ids.clone()
    labels[0, :prompt_len] = -100  

    outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss.item(), n_residues  

@torch.no_grad()
def compute_recovery(model, tokenizer, records, few_shot_pool,
                     modality, max_length, decimal_places):
    correct, total, examples = 0, 0, []
    is_instruct = modality.endswith("-inst")
    is_calpha   = modality.startswith("calpha")

    for r in tqdm(records, desc="Eval"):

        if is_instruct and is_calpha:
            prompt = build_prompt_instruct_calpha(r, few_shot_pool, tokenizer, decimal_places)
        elif is_instruct:
            prompt = build_prompt_instruct_3di(r, few_shot_pool, tokenizer)
        elif is_calpha:
            prompt = build_prompt_base_calpha(r, few_shot_pool, decimal_places)
        else:
            prompt = build_prompt_base_3di(r, few_shot_pool)

        add_special = not is_instruct
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_length,
            add_special_tokens=add_special,
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=r["length"] * 2 + 20,  
            do_sample=False, temperature=None, top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        gen_clean = gen.strip()
        if gen_clean.lower().startswith("<seq>"):
            gen_clean = gen_clean[len("<seq>"):].strip()

        pred = "".join(gen_clean.split())
        pred = "".join(c for c in pred if c in AA_VOCAB)[:r["length"]]
        ref  = r["seq"]

        c = sum(p == q for p, q in zip(pred, ref))
        correct += c; total += len(ref)

        nll, n_res = compute_perplexity_for_record(
            model, tokenizer, r, few_shot_pool,
            modality, max_length, decimal_places,
        )
        ppl = math.exp(nll) if not math.isnan(nll) else float("nan")

        examples.append({
            "name":       r["name"],
            "length":     len(ref),
            "pred_len":   len(pred),
            "ref":        ref,
            "pred":       pred,
            "recovery":   c / len(ref),
            "nll":        nll,
            "perplexity": ppl,
            "n_residues": n_res,
        })

    overall_recovery = correct / total if total else 0.0
    valid   = [e for e in examples if not math.isnan(e["nll"])]
    avg_ppl = (math.exp(sum(e["nll"] * e["n_residues"] for e in valid) /
                        sum(e["n_residues"] for e in valid))
               if valid else float("nan"))
    return overall_recovery, avg_ppl, examples

def _wavg_rec(sub):
    return (sum(e["recovery"] * e["length"] for e in sub) / sum(e["length"] for e in sub)
            if sub else float("nan"))

def _wavg_ppl(sub):
    valid = [e for e in sub if not math.isnan(e["nll"])]
    if not valid: return float("nan")
    return math.exp(sum(e["nll"] * e["n_residues"] for e in valid) /
                    sum(e["n_residues"] for e in valid))


def print_results(label, recovery, avg_ppl, examples):
    short  = [e for e in examples if e["length"] <= 100]
    medium = [e for e in examples if 100 < e["length"] <= 300]
    long_  = [e for e in examples if e["length"] > 300]
    exact  = sum(1 for e in examples if e["pred_len"] == e["length"])

    print(f"\n{'='*60}\n  {label}  (n={len(examples)})\n{'='*60}")
    print(f"  Recovery (all)     : {recovery*100:6.2f}%")
    print(f"  Recovery (<=100aa) : {_wavg_rec(short)*100:6.2f}%  (n={len(short)})")
    print(f"  Recovery (101-300) : {_wavg_rec(medium)*100:6.2f}%  (n={len(medium)})")
    print(f"  Recovery (>300aa)  : {_wavg_rec(long_)*100:6.2f}%  (n={len(long_)})")
    print(f"  Exact length       : {exact}/{len(examples)}")
    print()
    print(f"  Perplexity (all)   : {avg_ppl:8.3f}  [residue-level]")
    print(f"  Perplexity (<=100) : {_wavg_ppl(short):8.3f}  (n={len(short)})")
    print(f"  Perplexity(101-300): {_wavg_ppl(medium):8.3f}  (n={len(medium)})")
    print(f"  Perplexity (>300)  : {_wavg_ppl(long_):8.3f}  (n={len(long_)})")

    for e in examples[:3]:
        flag = "ok" if e["pred_len"] == e["length"] else f"len={e['pred_len']}"
        print(f"\n  {e['name']}  rec={e['recovery']:.3f}  ppl={e['perplexity']:.2f}  {flag}")
        print(f"    ref : {e['ref'][:70]}")
        print(f"    pred: {e['pred'][:70]}")

def main(args):
    random.seed(42)

    key = "tokens_3di" if args.modality.startswith("3di") else "ca_coords"

    records = [json.loads(l) for l in open(args.test_file) if l.strip()]
    records = [r for r in records if r.get(key) and r.get("seq")]
    if args.n_eval > 0:
        records = random.sample(records, min(args.n_eval, len(records)))
    print(f"Evaluating {len(records)} chains  [modality={args.modality}]")

    tokenizer = setup_tokenizer(args.model_name)
    model     = load_model(args.model_name, tokenizer)

    all_results = {}

    for k in args.shot:
        random.seed(42)
        few_shot_pool = sample_few_shot_pool(args.train_file, args.modality, k)

        label = f"{k}-shot  [{args.modality}]  ({args.model_name.split('/')[-1]})"

        recovery, avg_ppl, exs = compute_recovery(
            model, tokenizer, records, few_shot_pool,
            args.modality, args.max_length, args.decimal_places,
        )
        print_results(label, recovery, avg_ppl, exs)
        all_results[label] = {
            "shot":             k,
            "modality":         args.modality,
            "model":            args.model_name,
            "recovery":         recovery,
            "perplexity":       avg_ppl,
            "n_eval":           len(records),
            "fewshot_examples": [e["name"] for e in few_shot_pool],
            "examples":         exs[:10],
        }

    if len(all_results) > 1:
        print(f"\n{'Model / Setting':<50} {'Recovery':>10} {'Perplexity':>12}")
        print("-" * 76)
        for lbl, res in all_results.items():
            print(f"{lbl:<50} {res['recovery']*100:>9.2f}%  {res['perplexity']:>11.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(all_results, open(args.out, "w"), indent=2)
    print(f"\nSaved  {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Zero/few-shot eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modality", required=True,
                   choices=["3di", "3di-inst", "calpha", "calpha-inst"])
    p.add_argument("--shot", nargs="+", type=int, default=[0], metavar="K")
    p.add_argument("--model_name", default=None)
    p.add_argument("--test_file",  default="data/tokenized/test.jsonl")
    p.add_argument("--train_file", default="data/tokenized/train.jsonl")
    p.add_argument("--decimal_places", type=int, default=1)
    p.add_argument("--n_eval",     type=int, default=-1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--out",        default="results/eval_fewshot.json")
    args = p.parse_args()

    if args.model_name is None:
        args.model_name = (
            "meta-llama/Meta-Llama-3-8B-Instruct"
            if args.modality.endswith("-inst")
            else "meta-llama/Meta-Llama-3-8B"
        )

    main(args)