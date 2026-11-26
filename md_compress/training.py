"""Training loops: staged (side-chain stack on true backbones -> backbone codec -> end-to-end
fine-tuning) or purely end-to-end."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import losses as L
from .compression import MDCompressor


@dataclass
class LossWeights:
    backbone: float = 1.0
    sc_local: float = 1.0
    sc_global: float = 0.5
    chi: float = 1.0
    chi_norm: float = 0.02
    nll: float = 0.1
    clash: float = 0.1     # set 0 to disable the soft clash term
    aux_pass: float = 0.5  # weight of intermediate refinement passes


def prepare(data, device="cpu"):
    t = data.torch(device)
    from .geometry import SideChainBuilder
    chi, chi_mask = SideChainBuilder().to(device).chis_from_atom14(t["coords"], t["aatype"])
    t["chi"], t["chi_mask"] = chi, chi_mask[0]
    return t


def packer_loss(model: MDCompressor, outs, true_x, true_bb, true_chi, t, w: LossWeights, use_global=False):
    aat, m = t["aatype"], t["atom_mask"]
    total, logs = 0.0, {}
    for p, o in enumerate(outs):
        wp = 1.0 if p == len(outs) - 1 else w.aux_pass
        pred_bb = o["atom14"][:, :, :4]
        l_loc = L.sidechain_local_loss(o["atom14"], true_x, pred_bb, true_bb, aat, m)
        l_chi = L.chi_loss(o["chi_sc"], true_chi, aat, o["chi_mask"])
        lp = w.sc_local * l_loc + w.chi * l_chi
        if o.get("chi_raw_norm") is not None:
            lp = lp + w.chi_norm * (((o["chi_raw_norm"] - 1) ** 2) * o["chi_mask"]).mean()
        if o.get("nll") is not None:
            lp = lp + w.nll * o["nll"]
        if use_global:
            lp = lp + w.sc_global * L.sidechain_global_loss(o["atom14"], true_x, aat, m)
        total = total + wp * lp
        if p == len(outs) - 1:
            logs.update(sc_local=l_loc.item(), chi=l_chi.item())
    if w.clash > 0:
        lc = L.clash_loss(outs[-1]["atom14"], aat, m, t["chain_index"])
        total = total + w.clash * lc
        logs["clash"] = lc.item()
    return total, logs


def _batches(idx, bs, rng):
    idx = rng.permutation(idx)
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def train_sidechain_stage(model, t, train_idx, epochs=30, bs=8, lr=2e-3, w=LossWeights(), log=print, seed=0):
    """Stage 1: mapping net + side-chain model on TRUE backbones (teacher forcing)."""
    from .losses import canonicalize_chi
    opt = torch.optim.AdamW(model.packer.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        t0, agg = time.time(), []
        for b in _batches(train_idx, bs, rng):
            x = t["coords"][b]
            chi = t["chi"][b]
            chi_t = canonicalize_chi(chi, t["aatype"]) if model.packer.sidechain.head.probabilistic else None
            outs = model.pack(x[:, :, :4], t["aatype"], t["chain_index"], t["chain_break"], chi_true=chi_t)
            loss, logs = packer_loss(model, outs, x, x[:, :, :4], chi, t, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.packer.parameters(), 1.0)
            opt.step(); agg.append(dict(loss=loss.item(), **logs))
        sched.step()
        log(f"[sidechain] ep {ep+1}/{epochs} " + _fmt(agg) + f" ({time.time()-t0:.1f}s)")


def train_codec_stage(model, t, train_idx, epochs=200, bs=16, lr=2e-3, latent_l2=1e-4, log=print, seed=0):
    """Stage 2: backbone autoencoder (aligned frame)."""
    codec = model.codec
    opt = torch.optim.AdamW(codec.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    bb_all, _, _ = codec.align(t["coords"][:, :, :4])
    for ep in range(epochs):
        agg = []
        for b in _batches(train_idx, bs, rng):
            bb = bb_all[b]
            z = codec.encode(bb, t["aatype"])
            rec = codec.decode(z, t["aatype"])
            l_bb = L.backbone_loss(rec, bb)
            loss = l_bb + latent_l2 * (z ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            agg.append(dict(bb_mse=l_bb.item()))
        sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            log(f"[codec] ep {ep+1}/{epochs} " + _fmt(agg))


def train_end_to_end(model, t, train_idx, epochs=20, bs=8, lr=5e-4, w=LossWeights(), log=print, seed=0):
    """Stage 3 (or from scratch): latent -> backbone -> packer, all losses, all params."""
    from .losses import canonicalize_chi
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        t0, agg = time.time(), []
        for b in _batches(train_idx, bs, rng):
            x = t["coords"][b]
            bb_al, R_, t_ = model.codec.align(x[:, :, :4])
            x_al = (x.reshape(len(b), -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(x.shape)
            chi = t["chi"][b]
            chi_t = canonicalize_chi(chi, t["aatype"]) if model.packer.sidechain.head.probabilistic else None
            z = model.codec.encode(bb_al, t["aatype"])
            bb_hat, outs = model.decode_full(z, t["aatype"], t["chain_index"], t["chain_break"], chi_true=chi_t)
            l_bb = L.backbone_loss(bb_hat, bb_al)
            lp, logs = packer_loss(model, outs, x_al, bb_al, chi, t, w, use_global=True)
            loss = w.backbone * l_bb + lp
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); agg.append(dict(loss=loss.item(), bb_mse=l_bb.item(), **logs))
        sched.step()
        log(f"[e2e] ep {ep+1}/{epochs} " + _fmt(agg) + f" ({time.time()-t0:.1f}s)")


def _fmt(agg):
    keys = agg[0].keys()
    return " ".join(f"{k}={np.mean([a[k] for a in agg]):.4f}" for k in keys)


@torch.no_grad()
def evaluate_model(model, t, idx, mode="e2e", bs=8, n_refine=None):
    """mode='true_bb': side chains from the TRUE backbone (isolates packer quality);
    mode='e2e': full decompression from latents (codec + packer)."""
    model.eval()
    preds, chis, trues, tchis = [], [], [], []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x = t["coords"][b]
        bb_al, R_, t_ = model.codec.align(x[:, :, :4])
        x_al = (x.reshape(len(b), -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(x.shape)
        if mode == "true_bb":
            outs = model.pack(bb_al, t["aatype"], t["chain_index"], t["chain_break"], n_refine=n_refine)
        else:
            z = model.codec.encode(bb_al, t["aatype"])
            z = z.half().float()  # storage precision
            _, outs = model.decode_full(z, t["aatype"], t["chain_index"], t["chain_break"], n_refine=n_refine)
        preds.append(outs[-1]["atom14"]); chis.append(outs[-1]["chi_sc"])
        trues.append(x_al); tchis.append(t["chi"][b])
    model.train()
    return L.evaluate(torch.cat(preds), torch.cat(trues), torch.cat(chis), torch.cat(tchis), t["aatype"],
                      t["atom_mask"], t["chi_mask"], t["chain_index"])


def save_checkpoint(model: MDCompressor, path, config: dict):
    torch.save(dict(state_dict=model.state_dict(), config=config), path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = MDCompressor(cfg["n_res"], codec=cfg["codec"], codec_kw=cfg.get("codec_kw"),
                         packer_kw=cfg.get("packer_kw"), use_chi_prior=cfg.get("use_chi_prior", True))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg
