#!/usr/bin/env python
"""Evaluate shared multi-protein models on proteins never seen in training.

Sets: mdCATH test domains (superfamilies disjoint from train/val; replica 0 at 320 K, frames every
4 ns, up to 100 frames) and the 8 external systems of the CPU study (all frames).
For every protein only *stored per-topology data* is used: reference backbone (+ mean rotamers for
prior models), computed from the protein's own trajectory, as the compressor does.
Models: staged = shared codec + packer (S/L, +/- prior); e2e = jointly fine-tuned checkpoints.
Baselines: ideal-geometry floor (true chi), per-protein mean rotamers (= the stored chi prior alone),
one-rotamer-per-type library from the training set (no per-protein storage), MD clash reference.
Incremental: results.json is updated after every protein.
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Domain, bind, build_model, fit_buffers, load_npz_domain, load_weights,  # noqa: E402
                    type_mode_rotamers)
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/content/data/mdcath")
ap.add_argument("--external", default="/content/data/external")
ap.add_argument("--runs", default="/content/runs")
ap.add_argument("--out", default="/content/runs/eval/results.json")
ap.add_argument("--max_frames", type=int, default=100)
ap.add_argument("--only", default="", help="comma list of model-name prefixes to evaluate (e.g. staged)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
builder = SideChainBuilder().to(dev)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
man = json.load(open(f"{a.data}/manifest.json"))


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def save():
    json.dump(res, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)


def short(m):
    return dict(bb_rmsd=m["backbone_rmsd"], sc_rmsd=m["sidechain_rmsd"], all_heavy=m["all_heavy_rmsd"],
                chi_mae=m["chi_mae_deg"], chi1_acc40=m["chi1_acc_40deg"], clash=m["clash_per_1000_atoms"])


# ------------------------------------------------------------------ models
def ck(p):
    return torch.load(p, map_location=dev, weights_only=False)["weights"] if os.path.exists(p) else None


codec_w = ck(f"{a.runs}/codec_S_prior/best.pt")
MODELS = {}
for v in ("S", "L"):
    for prior in (True, False):
        tag = f"{v}_{'prior' if prior else 'noprior'}"
        pw = ck(f"{a.runs}/packer_{tag}/best.pt")
        if pw is not None and codec_w is not None:
            m = build_model(v, prior, dev)
            load_weights(m, codec_w, "codec."); load_weights(m, pw, "packer.")
            MODELS[f"staged_{tag}"] = m.eval()
        ew = ck(f"{a.runs}/e2e_{tag}/best.pt")
        if ew is not None:
            m = build_model(v, prior, dev)
            load_weights(m, ew)
            MODELS[f"e2e_{tag}"] = m.eval()
if a.only:
    MODELS = {k: v for k, v in MODELS.items() if any(k.startswith(p) for p in a.only.split(","))}
log(f"models: {list(MODELS)}")

# ------------------------------------------------------------------ type-mode rotamer library (training set)
train = sorted(d for d, v in man["domains"].items() if v["split"] == "train")
lib_path = f"{a.runs}/eval/type_mode_rotamers.pt"
if os.path.exists(lib_path):
    LIB = torch.load(lib_path)
else:
    chis, aats, cms = [], [], []
    for d in train:
        dom, _ = load_npz_domain(f"{a.data}/domains/{d}.npz", dev, max_frames=20, stride=10)
        c, cm = builder.chis_from_atom14(torch.as_tensor(dom.coords, device=dev), dom.aatype)
        chis.append(c); aats.append(dom.aatype); cms.append(cm[0])
    LIB = type_mode_rotamers(chis, aats, cms)
    torch.save(LIB, lib_path)
LIB = LIB.to(dev)

# ------------------------------------------------------------------ sequence identity to the training set
train_seqs = {d: str(np.load(f"{a.data}/domains/{d}.npz")["seq"]) for d in train}
try:
    from Bio import Align
    aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0, open_gap_score=-2,
                                    extend_gap_score=-0.5)

    def identity(s, t):
        al = aligner.align(s, t)[0]
        n = sum(sum(1 for i, j in zip(range(a0, a1), range(b0, b1)) if s[i] == t[j])
                for (a0, a1), (b0, b1) in zip(*al.aligned))
        return n / min(len(s), len(t))
except Exception:  # pragma: no cover
    identity = None


@torch.no_grad()
def run_model(m, dom, x, mode):
    bind(m, dom)
    out_p, out_c = [], []
    for s in range(0, len(x), 16):
        xb = x[s:s + 16]
        bb_al, R_, t_ = m.codec.align(xb[:, :, :4])
        if mode == "true_bb":
            outs = m.pack(bb_al, dom.aatype, dom.chain_index, dom.chain_break)
        else:
            z = m.codec.encode(bb_al, dom.aatype).half().float()  # storage precision
            _, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        out_p.append(m.codec.unalign(outs[-1]["atom14"], R_, t_)); out_c.append(outs[-1]["chi_sc"])
    return torch.cat(out_p), torch.cat(out_c)


def evaluate_system(key, dom, x_all, x, meta):
    t0 = time.time()
    old = res.get(key)
    fit_buffers(dom, x_all, builder)  # stored per-topology data from the protein's own trajectory
    chi, _ = builder.chis_from_atom14(x, dom.aatype)
    ev = lambda p, c: short(L.evaluate(p, x, c, chi, dom.aatype, dom.atom_mask, dom.chi_mask, dom.chain_index))
    if old is not None:
        r = old
        for name, m in MODELS.items():
            if name not in r:
                r[name] = dict(e2e=ev(*run_model(m, dom, x, "e2e")))
                if name.startswith("staged"):
                    r[name]["true_bb"] = ev(*run_model(m, dom, x, "true_bb"))
        res[key] = r; save(); log(f"{key}: added models"); return
    r = dict(meta, n_res=dom.R, n_heavy=dom.n_heavy, n_eval_frames=len(x))
    rec, _ = builder(x[:, :, :4], dom.aatype, chi)
    r["floor"] = ev(rec, chi)
    mc = dom.chi_ref.expand_as(chi)
    rec, _ = builder(x[:, :, :4], dom.aatype, mc)
    r["static_rotamers"] = ev(rec, mc)
    lc = (LIB[dom.aatype] * dom.chi_mask[..., None]).expand_as(chi)
    rec, _ = builder(x[:, :, :4], dom.aatype, lc)
    r["type_library"] = ev(rec, lc)
    r["md_clash"] = L.clash_score(x, dom.aatype, dom.atom_mask, dom.chain_index)
    raw, stored = dom.n_heavy * 12, dom.R * 4 * 2 + 7 * 4
    r["compression"] = dict(raw_bytes_per_frame=raw, stored_bytes_per_frame=stored, ratio=raw / stored,
                            per_protein_ref_backbone_bytes=dom.R * 12 * 4, per_protein_chi_prior_bytes=dom.R * 8 * 4)
    for name, m in MODELS.items():
        r[name] = dict(e2e=ev(*run_model(m, dom, x, "e2e")))
        if name.startswith("staged"):
            r[name]["true_bb"] = ev(*run_model(m, dom, x, "true_bb"))
    if identity is not None and dom.seq:
        ids = [identity(dom.seq, s) for s in train_seqs.values()]
        r["max_seq_identity_to_train"] = float(max(ids))
    r["seconds"] = time.time() - t0
    res[key] = r
    save()
    best = {n: round(r[n]["e2e"]["sc_rmsd"], 3) for n in MODELS}
    log(f"{key} ({dom.R} res): floor {r['floor']['sc_rmsd']:.3f} static {r['static_rotamers']['sc_rmsd']:.3f} "
        f"lib {r['type_library']['sc_rmsd']:.3f} | {best}")


test = sorted(d for d, v in man["domains"].items() if v["split"] == "test")
for d in test:
    key = f"mdcath/{d}"
    if key in res and all(n in res[key] for n in MODELS):
        continue
    z = np.load(f"{a.data}/domains/{d}.npz")
    rep0 = np.where(z["replica"] == 0)[0]
    x_all = torch.as_tensor(z["coords"][rep0].astype(np.float32), device=dev)
    sel = np.arange(0, len(rep0), 2)[:a.max_frames]
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], dev)
    dom.seq = str(z["seq"])
    evaluate_system(key, dom, x_all, x_all[sel], dict(set="mdcath_test", sf=man["domains"][d]["sf"]))
for p in sorted(glob.glob(f"{a.external}/*.npz")):
    name = os.path.basename(p)[:-4]
    key = f"external/{name}"
    if key in res and all(n in res[key] for n in MODELS):
        continue
    dom, z = load_npz_domain(p, dev)
    x_all = torch.as_tensor(dom.coords, device=dev)
    evaluate_system(key, dom, x_all, x_all, dict(set="external"))
log("eval done")
