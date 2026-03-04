import json, math, os, argparse, subprocess, tempfile
from pathlib import Path

BACKBONE = ["N", "CA", "C", "O"]
AA_3 = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
    'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
    'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR',
    'X':'UNK','U':'UNK','O':'UNK','B':'ASN','Z':'GLN','J':'LEU',
}

def is_nan(v):
    try: return v is None or math.isnan(float(v))
    except: return True

def residue_nan(coords, i):
    for atom in BACKBONE:
        xyz = coords.get(atom, [])[i] if i < len(coords.get(atom, [])) else None
        if xyz is None or len(xyz) != 3 or any(is_nan(v) for v in xyz): return True
    return False

def trim(seq, coords, min_len):
    nan_mask = [residue_nan(coords, i) for i in range(len(seq))]
    start = next((i for i, bad in enumerate(nan_mask) if not bad), None)
    if start is None: return None
    end = len(seq) - next(i for i, bad in enumerate(reversed(nan_mask)) if not bad)
    if any(nan_mask[start:end]): return None   # internal gap
    if end - start < min_len: return None
    return seq[start:end], {a: coords[a][start:end] for a in BACKBONE}

def write_pdb(name, seq, coords, path):
    lines, idx = [], 1
    for ri, aa in enumerate(seq):
        for atom in BACKBONE:
            x, y, z = [float(v) for v in coords[atom][ri]]
            lines.append(
                f"ATOM  {idx:5d}  {atom:<3s} {AA_3.get(aa,'UNK')} A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom[0]:>2s}\n"
            )
            idx += 1
    lines.append("END\n")
    Path(path).write_text("".join(lines))

def run_foldseek(pdb_dir, binary, threads):
    with tempfile.TemporaryDirectory() as tmp:
        db, fasta = f"{tmp}/db", f"{tmp}/3di.fasta"
        for cmd in [
            [binary, "createdb", pdb_dir, db, "--threads", str(threads)],
            [binary, "lndb", db + "_h", db + "_ss_h"],
            [binary, "convert2fasta", db + "_ss", fasta],
        ]:
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"Foldseek failed: {r.stderr.decode()[:400]}")
        tokens, name, parts = {}, None, []
        for line in Path(fasta).read_text().splitlines():
            if line.startswith(">"):
                if name: tokens[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
        if name: tokens[name] = "".join(parts)
    return tokens

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

    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for name, (seq, coords) in clean.items():
        p = pdb_dir / f"{name}.pdb"
        if not p.exists():
            write_pdb(name, seq, coords, p)

    print("Running Foldseek...")
    tokens = run_foldseek(str(pdb_dir), args.foldseek_bin, args.threads)
    print(f"Foldseek: {len(tokens)} entries")

    for split, ids in splits.items():
        records = []
        for name in ids:
            if name not in clean or name not in tokens: continue
            seq, tok = clean[name][0], tokens[name]
            if len(tok) != len(seq): continue
            records.append({"name": name, "seq": seq, "length": len(seq), "tokens_3di": tok})
        (out_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"  {split:12s}: {len(records)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data/cath3")
    p.add_argument("--out_dir",      default="data/tokenizedCA")
    p.add_argument("--foldseek_bin", default="foldseek")
    p.add_argument("--min_len",      type=int, default=30)
    p.add_argument("--threads",      type=int, default=4)
    main(p.parse_args())