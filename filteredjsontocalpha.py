import json, math, os, argparse
from pathlib import Path

BACKBONE = ["N", "CA", "C", "O"]

def is_nan(v):
    try: return v is None or math.isnan(float(v))
    except: return True

def residue_nan(coords, i):
    for atom in BACKBONE:
        xyz = coords.get(atom, [])[i] if i < len(coords.get(atom, [])) else None
        if xyz is None or len(xyz) != 3 or any(is_nan(v) for v in xyz):
            return True
    return False

def get_ca(coords, i):
    xyz = coords["CA"][i]
    return [float(v) for v in xyz]

def trim(seq, coords, min_len):
    nan_mask = [residue_nan(coords, i) for i in range(len(seq))]
    start = next((i for i, bad in enumerate(nan_mask) if not bad), None)
    if start is None: return None
    end = len(seq) - next(i for i, bad in enumerate(reversed(nan_mask)) if not bad)
    if any(nan_mask[start:end]): return None   # internal gap
    if end - start < min_len: return None
    trimmed_coords = {a: coords[a][start:end] for a in BACKBONE}
    return seq[start:end], trimmed_coords

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw    = [json.loads(l) for l in Path(args.data_dir, "chain_set.jsonl").open() if l.strip()]
    splits = {k: set(v) for k, v in
              json.load(open(Path(args.data_dir, "chain_set_splits.json"))).items()
              if k in {"train", "validation", "test"}}
    print(f"Loaded {len(raw)} entries | splits: { {k: len(v) for k, v in splits.items()} }")

    clean = {}
    for e in raw:
        name, seq, coords = e["name"], e["seq"], e["coords"]
        if not any(residue_nan(coords, i) for i in range(len(seq))):
            if len(seq) >= args.min_len:
                clean[name] = (seq, coords)
        else:
            result = trim(seq, coords, args.min_len)
            if result:
                clean[name] = result
    print(f"Kept {len(clean)}/{len(raw)} after NaN filtering (min_len={args.min_len})")

    for split, ids in splits.items():
        records = []
        for name in ids:
            if name not in clean: continue
            seq, coords = clean[name]
            ca_coords = [get_ca(coords, i) for i in range(len(seq))]
            records.append({
                "name":      name,
                "seq":       seq,
                "length":    len(seq),
                "ca_coords": ca_coords,  
            })
        (out_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        print(f"  {split:12s}: {len(records)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/cath2")
    p.add_argument("--out_dir",   default="data/tokenized_ca")
    p.add_argument("--min_len",   type=int, default=30)
    main(p.parse_args())