#!/usr/bin/env python
"""Multi-protein GPU training on mdCATH (shared weights, per-protein stored buffers).

Stages (run as separate durable processes; each is time-budgeted and resumable):
  buffers : per-protein reference backbone / scale / chi prior for every domain -> buffers.pt (run once)
  codec   : shared per-residue backbone codec (latent 4 / residue)
  packer  : mapping net + side-chain model on TRUE backbones (teacher forcing), variant S or L, +/- chi prior
  e2e     : latent -> backbone -> packer, all weights, initialised from a codec + a packer checkpoint

The learning rate follows a cosine over the *time budget* (fraction of --minutes elapsed), so a run
can be resumed after an interruption from <run>/last.pt and still finish on schedule.
Validation (held-out superfamilies) every --eval_every seconds; best.pt keeps the best val score.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Domain, bind, build_model, fit_buffers, load_weights, weights  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder  # noqa: E402
from md_compress.memory import param_table  # noqa: E402
from md_compress.training import LossWeights, packer_loss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["buffers", "codec", "packer", "e2e"])
ap.add_argument("--data", default="/content/data/mdcath")
ap.add_argument("--runs", default="/content/runs")
ap.add_argument("--name", default=None)
ap.add_argument("--variant", default="S")
ap.add_argument("--no_prior", action="store_true")
ap.add_argument("--minutes", type=float, default=30)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--clash", type=float, default=0.1)
ap.add_argument("--eval_every", type=float, default=180)
ap.add_argument("--init_codec", default=None)
ap.add_argument("--init_packer", default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)  # several runs share the VM CPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
man = json.load(open(f"{a.data}/manifest.json"))
index = json.load(open(f"{a.data}/train_pack/index.json"))
mm = np.load(f"{a.data}/train_pack/coords.npy", mmap_mode="r")
builder = SideChainBuilder().to(dev)
BUF = f"{a.data}/buffers.pt"


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def meta(d):
    return np.load(f"{a.data}/domains/{d}.npz")


# ------------------------------------------------------------------ domains
train_names = sorted(index)
val_names = sorted(d for d, v in man["domains"].items() if v["split"] == "val")
doms = {}
for d in train_names:
    z = meta(d)
    doms[d] = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], dev,
                     off=index[d]["off"], T=index[d]["T"])
VAL_FRAMES = 16
for d in val_names:
    z = meta(d)
    x = z["coords"][z["replica"] == 0]
    sel = np.linspace(0, len(x) - 1, VAL_FRAMES).astype(int)
    doms[d] = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], dev,
                     coords=x[sel].astype(np.float32))
    doms[d].all_coords = z["coords"]

if a.stage == "buffers" or not os.path.exists(BUF) or set(doms) - set(torch.load(BUF)):
    t0 = time.time()
    buf = {}
    for d, dom in doms.items():
        if dom.coords is None:
            x = dom.frames(np.arange(dom.T), mm)
        else:
            x = torch.as_tensor(dom.all_coords.astype(np.float32), device=dev)  # whole trajectory of the val protein
        fit_buffers(dom, x, builder)
        buf[d] = dict(ref=dom.ref.cpu(), scale=dom.scale.cpu(), chi_ref=dom.chi_ref.cpu(), chi_mask=dom.chi_mask.cpu())
    torch.save(buf, BUF + ".tmp")
    os.replace(BUF + ".tmp", BUF)
    log(f"buffers for {len(buf)} domains in {time.time()-t0:.0f}s")
    if a.stage == "buffers":
        sys.exit(0)
buf = torch.load(BUF)
for d, dom in doms.items():
    b = buf[d]
    dom.ref, dom.scale, dom.chi_ref, dom.chi_mask = (b["ref"].to(dev), b["scale"].to(dev), b["chi_ref"].to(dev),
                                                    b["chi_mask"].to(dev))

# ------------------------------------------------------------------ model / optimiser
prior = not a.no_prior
name = a.name or f"{a.stage}_{a.variant}_{'prior' if prior else 'noprior'}"
run = f"{a.runs}/{name}"
os.makedirs(run, exist_ok=True)
torch.manual_seed(a.seed)
model = build_model(a.variant, prior, dev)
if a.init_codec:
    load_weights(model, torch.load(a.init_codec, map_location=dev)["weights"], prefix="codec.")
if a.init_packer:
    load_weights(model, torch.load(a.init_packer, map_location=dev)["weights"], prefix="packer.")
if a.stage == "codec":
    params, lr = list(model.codec.parameters()), a.lr or 2e-3
elif a.stage == "packer":
    params, lr = list(model.packer.parameters()), a.lr or 1e-3
else:
    params, lr = list(model.parameters()), a.lr or 3e-4
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if a.stage == "codec" else 1e-4)
W = LossWeights(clash=a.clash)
state = dict(step=0, elapsed=0.0, best=float("inf"), history=[])
if os.path.exists(f"{run}/last.pt"):
    ck = torch.load(f"{run}/last.pt", map_location=dev, weights_only=False)
    load_weights(model, ck["weights"])
    opt.load_state_dict(ck["opt"])
    state = ck["state"]
    log(f"resumed {name} at step {state['step']} ({state['elapsed']/60:.1f} min)")
if state.get("done"):
    log("already done"); sys.exit(0)
cfg = dict(stage=a.stage, variant=a.variant, prior=prior, bs=a.bs, lr=lr, minutes=a.minutes, clash=a.clash,
           params=param_table(model), n_train_domains=len(train_names), n_val_domains=len(val_names),
           init_codec=a.init_codec, init_packer=a.init_packer, gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu")
json.dump(cfg, open(f"{run}/config.json", "w"), indent=1)
log(f"{name}: params {cfg['params']}")


def save(path):
    torch.save(dict(weights=weights(model), opt=opt.state_dict(), state=state, config=cfg), path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def validate():
    model.eval()
    agg = []
    for d in val_names:
        dom = doms[d]
        bind(model, dom)
        x = dom.frames(np.arange(dom.T))
        chi, _ = builder.chis_from_atom14(x, dom.aatype)
        bb_al, R_, t_ = model.codec.align(x[:, :, :4])
        x_al = (x.reshape(len(x), -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(x.shape)
        if a.stage == "codec":
            z = model.codec.encode(bb_al, dom.aatype).half().float()
            rec = model.codec.decode(z, dom.aatype)
            agg.append(dict(bb_rmsd=((rec - bb_al) ** 2).sum(-1).mean().sqrt().item()))
            continue
        if a.stage == "packer":
            outs = model.pack(bb_al, dom.aatype, dom.chain_index, dom.chain_break)
        else:
            z = model.codec.encode(bb_al, dom.aatype).half().float()
            _, outs = model.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        m = L.evaluate(outs[-1]["atom14"], x_al, outs[-1]["chi_sc"], chi, dom.aatype, dom.atom_mask, dom.chi_mask,
                       dom.chain_index)
        agg.append(dict(bb_rmsd=m["backbone_rmsd"], sc_rmsd=m["sidechain_rmsd"], chi1=m["chi_mae_deg"][0],
                        chi1_acc40=m["chi1_acc_40deg"], clash=m["clash_per_1000_atoms"]))
    model.train()
    res = {k: float(np.mean([g[k] for g in agg])) for k in agg[0]}
    score = res["bb_rmsd"] if a.stage == "codec" else res["sc_rmsd"] + (res["bb_rmsd"] if a.stage == "e2e" else 0)
    return res, score


rng = np.random.default_rng(a.seed + state["step"])
budget = a.minutes * 60
t_last = time.time()
last_eval = state["elapsed"]
agg = []
model.train()
while state["elapsed"] < budget:
    frac = state["elapsed"] / budget
    cur_lr = lr * (min(1.0, (state["step"] + 1) / 200) * 0.5 * (1 + math.cos(math.pi * frac)) + 0.02)
    for g in opt.param_groups:
        g["lr"] = cur_lr
    dom = doms[train_names[rng.integers(len(train_names))]]
    bind(model, dom)
    idx = rng.choice(dom.T, min(a.bs, dom.T), replace=False)
    x = dom.frames(idx, mm)
    if a.stage == "codec":
        bb, _, _ = model.codec.align(x[:, :, :4])
        z = model.codec.encode(bb, dom.aatype)
        l_bb = L.backbone_loss(model.codec.decode(z, dom.aatype), bb)
        loss, logs = l_bb + 1e-4 * (z ** 2).mean(), dict(bb_mse=l_bb.item())
    else:
        chi, _ = builder.chis_from_atom14(x, dom.aatype)
        t = dict(aatype=dom.aatype, atom_mask=dom.atom_mask, chain_index=dom.chain_index)
        if a.stage == "packer":
            outs = model.pack(x[:, :, :4], dom.aatype, dom.chain_index, dom.chain_break)
            loss, logs = packer_loss(model, outs, x, x[:, :, :4], chi, t, W)
        else:
            bb_al, R_, t_ = model.codec.align(x[:, :, :4])
            x_al = (x.reshape(len(idx), -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(x.shape)
            z = model.codec.encode(bb_al, dom.aatype)
            bb_hat, outs = model.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
            l_bb = L.backbone_loss(bb_hat, bb_al)
            lp, logs = packer_loss(model, outs, x_al, bb_al, chi, t, W, use_global=True)
            loss = l_bb + lp
            logs["bb_mse"] = l_bb.item()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    agg.append(dict(loss=loss.item(), **logs))
    state["step"] += 1
    now = time.time()
    state["elapsed"] += now - t_last
    t_last = now
    if state["elapsed"] - last_eval >= a.eval_every or state["elapsed"] >= budget:
        last_eval = state["elapsed"]
        res, score = validate()
        t_last = time.time()  # validation time is not counted in the training budget
        tr = {k: float(np.mean([g[k] for g in agg])) for k in agg[0]}
        agg = []
        state["history"].append(dict(step=state["step"], minutes=state["elapsed"] / 60, lr=cur_lr, train=tr, val=res))
        if score < state["best"]:
            state["best"] = score
            save(f"{run}/best.pt")
        save(f"{run}/last.pt")
        json.dump(state["history"], open(f"{run}/history.json", "w"), indent=1)
        log(f"{name} step {state['step']} {state['elapsed']/60:.1f}/{a.minutes:.0f} min lr {cur_lr:.2e} "
            f"train {json.dumps({k: round(v, 4) for k, v in tr.items()})} val {json.dumps({k: round(v, 4) for k, v in res.items()})}")
state["done"] = True
save(f"{run}/last.pt")
log(f"{name} done: {state['step']} steps, best val {state['best']:.4f}")
