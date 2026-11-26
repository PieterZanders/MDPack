#!/usr/bin/env python
"""Multi-system experiments (CPU, incremental: results_multi/state.json is updated after every unit,
finished units are skipped on restart).

(a) per-protein interpolation: one model per system (main config, short schedule), interleaved split.
(b) held-out proteins: 4 folds, each trains ONE shared packer (with and without chi prior) and ONE
    shared per-residue backbone codec on 6 systems, and tests on the 2 unseen systems. For an unseen
    system only *stored per-topology data* is used: its reference backbone and (prior variant) its
    mean rotamers, both computed from its own trajectory, as the compressor would at compression time.
Also per system: ideal-geometry floor (true chi), fixed-average-rotamer baseline, MD clash reference.
"""
import json, os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
from md_compress import losses as L
from md_compress.compression import MDCompressor, compression_report
from md_compress.data import load_trajectory, split_frames
from md_compress.geometry import SideChainBuilder
from md_compress.memory import param_table
from md_compress.training import (LossWeights, evaluate_model, packer_loss, prepare, train_codec_stage,
                                  train_end_to_end, train_sidechain_stage)

torch.set_num_threads(8)
torch.set_flush_denormal(True)  # avoid CPU slowdowns from denormal activations
OUT = "results_multi_fast" if "--fast" in sys.argv else "results_multi"
os.makedirs(OUT, exist_ok=True)
STATE = f"{OUT}/state.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
FAST = "--fast" in sys.argv


def save():
    json.dump(state, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)
    with open(f"{OUT}/log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")


ATLAS = "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{0}/{0}_protein.zip"
SYSTEMS = {
    "ADK_4AKE": dict(top="data/adk.psf", traj="data/adk_dims.dcd", stride=1, desc="adenylate kinase, CHARMM DIMS open->closed",
                     source="MDAnalysisTests adk.psf/adk_dims.dcd"),
}
for pid, desc in [("1pch_A", "HPr phosphocarrier protein"), ("3cjs_B", "50S ribosomal protein L11 fragment"),
                  ("2ygs_A", "APAF1 CARD domain"), ("3czz_B", "cyanovirin-N"), ("3fp5_A", "acyl-CoA binding protein"),
                  ("2fip_A", "late genes activator p4"), ("3klr_A", "glycine cleavage system H protein")]:
    d = f"data/atlas/{pid}"
    if os.path.exists(f"{d}/{pid}_prod_R1_fit.xtc"):
        SYSTEMS[pid] = dict(top=f"{d}/{pid}.pdb", traj=f"{d}/{pid}_prod_R1_fit.xtc", stride=100,
                            desc=desc + ", ATLAS CHARMM36m 100 ns replica 1", source=ATLAS.format(pid))

EP = dict(sc=4, codec=100, e2e=1) if FAST else dict(sc=30, codec=600, e2e=8)
EP_FOLD = dict(sc=2, codec=50) if FAST else dict(sc=12, codec=400)
W = LossWeights(clash=0.1)


def short(m):
    return dict(bb_rmsd=m["backbone_rmsd"], sc_rmsd=m["sidechain_rmsd"], sc_local=m["sidechain_local_rmsd"],
                all_heavy=m["all_heavy_rmsd"], chi1_mae=m["chi_mae_deg"][0], chi_mae=m["chi_mae_deg"],
                chi1_acc40=m["chi1_acc_40deg"], clash=m["clash_per_1000_atoms"])


# ------------------------------------------------------------------ load all systems
DATA, T = {}, {}
for name, s in SYSTEMS.items():
    d = load_trajectory(s["top"], s["traj"], stride=s["stride"])
    DATA[name], T[name] = d, prepare(d)
    info = state.setdefault("systems", {}).setdefault(name, {})
    info.update(desc=s["desc"], source=s["source"], n_res=d.n_res, n_heavy=d.n_heavy_atoms(), n_frames=d.n_frames,
                stride=s["stride"], sequence_types=int(len(set(d.aatype.tolist()))))
save()
log("systems: " + ", ".join(f"{k}({v['n_res']} res, {v['n_frames']} fr)" for k, v in state["systems"].items()))

# ------------------------------------------------------------------ per-system references
b = SideChainBuilder()
for name in SYSTEMS:
    info = state["systems"][name]
    if "floor" in info:
        continue
    t = T[name]
    tr, te = split_frames(DATA[name].n_frames)
    x = t["coords"][te]
    rec, _ = b(x[:, :, :4], t["aatype"], t["chi"][te])
    info["floor"] = short(L.evaluate(rec, x, t["chi"][te], t["chi"][te], t["aatype"], t["atom_mask"], t["chi_mask"], t["chain_index"]))
    mc = torch.nn.functional.normalize(L.canonicalize_chi(t["chi"][tr], t["aatype"]).mean(0), dim=-1) * t["chi_mask"][..., None]
    mc = mc.expand_as(t["chi"][te])
    rec, _ = b(x[:, :, :4], t["aatype"], mc)
    info["static_rotamers"] = short(L.evaluate(rec, x, mc, t["chi"][te], t["aatype"], t["atom_mask"], t["chi_mask"], t["chain_index"]))
    info["md_clash"] = L.clash_score(x, t["aatype"], t["atom_mask"], t["chain_index"])
    save()

# ------------------------------------------------------------------ (a) per-protein interpolation
for name in SYSTEMS:
    key = f"per_protein/{name}"
    if key in state:
        continue
    torch.manual_seed(0)
    d, t = DATA[name], T[name]
    tr, te = split_frames(d.n_frames)
    m = MDCompressor(d.n_res, codec_kw=dict(latent_dim=4), packer_kw=dict(n_refine=2))
    m.codec.fit_reference(t["coords"][tr][:, :, :4])
    m.fit_chi_prior(t["chi"][tr], t["chi_mask"], t["aatype"])
    t0 = time.time()
    q = lambda s: None
    train_sidechain_stage(m, t, tr, epochs=EP["sc"], w=W, log=q)
    train_codec_stage(m, t, tr, epochs=EP["codec"], log=q)
    train_end_to_end(m, t, tr, epochs=EP["e2e"], w=W, log=q)
    comp = compression_report(d.n_heavy_atoms(), m.codec, d.n_frames, param_table(m)["total"])
    state[key] = dict(e2e=short(evaluate_model(m, t, te, "e2e")), true_bb=short(evaluate_model(m, t, te, "true_bb")),
                      train_s=time.time() - t0, compression=comp)
    save()
    log(f"(a) {name}: " + json.dumps(state[key]["e2e"]))

# ------------------------------------------------------------------ (b) held-out folds
names = list(SYSTEMS)
rng = np.random.default_rng(0)
order = [names[i] for i in rng.permutation(len(names))]
n_folds = 4
folds = [order[i::n_folds] for i in range(n_folds)]
state["folds"] = folds
save()


def make_models(prior, shared=None):
    """One MDCompressor per system sharing all weights (per-system buffers: reference backbone, scale,
    chi prior)."""
    models = {}
    for name in names:
        d, t = DATA[name], T[name]
        m = MDCompressor(d.n_res, codec_kw=dict(latent_dim=4), packer_kw=dict(n_refine=2), use_chi_prior=prior)
        if shared is None:
            shared = dict(packer=m.packer, enc=m.codec.enc, dec=m.codec.dec, aa=m.codec.aa)
        m.packer, m.codec.enc, m.codec.dec, m.codec.aa = shared["packer"], shared["enc"], shared["dec"], shared["aa"]
        # per-topology stored data, computed from the system's own trajectory (all frames)
        m.codec.fit_reference(t["coords"][:, :, :4])
        m.fit_chi_prior(t["chi"], t["chi_mask"], t["aatype"])
        models[name] = m
    return models, shared


def train_packer_multi(models, shared, train_names, epochs, bs=8, n_sub=40):
    opt = torch.optim.AdamW(shared["packer"].parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(1)
    for ep in range(epochs):
        batches = []
        for n in train_names:
            tr, _ = split_frames(DATA[n].n_frames)
            tr = rng.choice(tr, min(n_sub, len(tr)), replace=False)
            batches += [(n, tr[i:i + bs]) for i in range(0, len(tr), bs)]
        rng.shuffle(batches)
        tot = []
        for n, bidx in batches:
            m, t = models[n], T[n]
            x = t["coords"][bidx]
            outs = m.pack(x[:, :, :4], t["aatype"], t["chain_index"], t["chain_break"])
            loss, _ = packer_loss(m, outs, x, x[:, :, :4], t["chi"][bidx], t, W)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(shared["packer"].parameters(), 1.0)
            opt.step(); tot.append(loss.item())
        sched.step()
    return float(np.mean(tot))


def train_codec_multi(models, shared, train_names, epochs, bs=16):
    params = list(shared["enc"].parameters()) + list(shared["dec"].parameters()) + list(shared["aa"].parameters())
    opt = torch.optim.AdamW(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(2)
    al = {}
    for n in train_names:
        al[n] = models[n].codec.align(T[n]["coords"][:, :, :4])[0]
    for ep in range(epochs):
        batches = []
        for n in train_names:
            tr, _ = split_frames(DATA[n].n_frames)
            tr = rng.permutation(tr)
            batches += [(n, tr[i:i + bs]) for i in range(0, len(tr), bs)]
        rng.shuffle(batches)
        for n, bidx in batches:
            c = models[n].codec
            bb = al[n][bidx]
            z = c.encode(bb, T[n]["aatype"])
            loss = L.backbone_loss(c.decode(z, T[n]["aatype"]), bb) + 1e-4 * (z ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


for fi, test_names in enumerate(folds):
    key = f"fold{fi}"
    if not test_names:
        continue
    if key in state and state[key].get("done"):
        continue
    train_names = [n for n in names if n not in test_names]
    log(f"(b) fold {fi}: train {train_names} test {test_names}")
    fold = state.setdefault(key, dict(train=train_names, test=test_names))
    t0 = time.time()
    torch.manual_seed(fi)
    models_p, shared_p = make_models(prior=True)
    models_n, shared_n = make_models(prior=False)
    # one codec shared by both packer variants
    shared_n["enc"], shared_n["dec"], shared_n["aa"] = shared_p["enc"], shared_p["dec"], shared_p["aa"]
    for n in names:
        models_n[n].codec.enc, models_n[n].codec.dec, models_n[n].codec.aa = shared_p["enc"], shared_p["dec"], shared_p["aa"]
    train_codec_multi(models_p, shared_p, train_names, EP_FOLD["codec"])
    log(f"  codec done ({time.time()-t0:.0f}s)")
    train_packer_multi(models_p, shared_p, train_names, EP_FOLD["sc"])
    log(f"  prior packer done ({time.time()-t0:.0f}s)")
    train_packer_multi(models_n, shared_n, train_names, EP_FOLD["sc"])
    log(f"  no-prior packer done ({time.time()-t0:.0f}s)")
    for n in test_names + train_names[:2]:
        _, te = split_frames(DATA[n].n_frames)
        r = {}
        for tag, ms in (("prior", models_p), ("noprior", models_n)):
            r[tag] = dict(e2e=short(evaluate_model(ms[n], T[n], te, "e2e")),
                          true_bb=short(evaluate_model(ms[n], T[n], te, "true_bb")))
        r["held_out"] = n in test_names
        r["compression_prior"] = compression_report(DATA[n].n_heavy_atoms(), models_p[n].codec, DATA[n].n_frames,
                                                    param_table(models_p[n])["total"])
        fold.setdefault("eval", {})[n] = r
        save()
        log(f"  {n} ({'held-out' if n in test_names else 'train'}): prior {json.dumps(r['prior']['e2e'])} | "
            f"noprior {json.dumps(r['noprior']['e2e'])}")
    fold["done"] = True
    fold["time_s"] = time.time() - t0
    save()
log("all done")
