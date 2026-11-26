"""Rotamer-state and per-residue metric helpers, copied verbatim from eval_stability.py (that script runs on import)."""
import numpy as np, torch
from md_compress import residue_constants as rc
from md_compress import losses as L
from md_compress.geometry import frames_from_backbone, kabsch, to_local
CORE, PRO_CORE, OCC_STABLE, OCC_UNSTABLE, MAX_STABLE_RATE, BURIED_NB = 40.0, 10.0, 0.90, 0.70, 0.02, 16
SP3_CHI2 = {rc.RESTYPE_ORDER[r] for r in ("ARG", "GLN", "GLU", "ILE", "LEU", "LYS", "MET")}
PRO = rc.RESTYPE_ORDER["PRO"]

# ------------------------------------------------------------------ rotamer states
def ang(sc):  # (sin,cos) -> degrees in (-180,180]
    return np.degrees(np.arctan2(sc[..., 0], sc[..., 1]))


def nearest_well(chi1, aat):
    """[T,R] chi1 deg -> nearest well index (0 p, 1 t, 2 m; PRO 0/1)."""
    d = np.stack([np.abs((chi1 - c + 180) % 360 - 180) for c in (60, 180, -60)], -1)
    w = d.argmin(-1)
    pro = aat == PRO
    w[:, pro] = (chi1[:, pro] < 0).astype(int)
    return w


def hysteresis_states(chi, aat, chi_mask):
    """chi [T,R,4] deg -> joint state [T,R] with core-to-core assignment."""
    T, R = chi.shape[:2]
    out = np.zeros((T, R), int)
    for k, use in ((0, np.ones(R, bool)), (1, np.isin(aat, list(SP3_CHI2)) & (chi_mask[:, 1] > 0))):
        x = chi[:, :, k]
        d = np.stack([np.abs((x - c + 180) % 360 - 180) for c in (60, 180, -60)], -1)
        core = np.where(d.min(-1) <= CORE, d.argmin(-1), -1)
        near = d.argmin(-1)
        if k == 0:
            pro = aat == PRO
            core[:, pro] = np.where(np.abs(x[:, pro]) >= PRO_CORE, (x[:, pro] < 0).astype(int), -1)
            near[:, pro] = (x[:, pro] < 0).astype(int)
        st = np.zeros((T, R), int)
        st[0] = np.where(core[0] >= 0, core[0], near[0])
        for t in range(1, T):
            st[t] = np.where(core[t] >= 0, core[t], st[t - 1])
        out = out * 3 + np.where(use[None], st, 0)
    return out


def occ_and_rate(states, dt):
    T = states.shape[0]
    occ = np.array([np.bincount(states[:, i]).max() / T for i in range(states.shape[1])])
    trans = (states[1:] != states[:-1]).sum(0)
    return occ, trans / max((T - 1) * dt, 1e-9)


def jsd(p, q):
    m = 0.5 * (p + q)
    f = lambda a, b: np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(-1)
    return 0.5 * f(p, m) + 0.5 * f(q, m)


def pops(w, n=3):
    return np.stack([(w == k).mean(0) for k in range(n)], -1)  # [R,n]


# ------------------------------------------------------------------ per-residue metrics for one prediction
SWAP = None


def canon_local(xl, ref, aat):
    """choose per frame the symmetric swap closest to ref (local coords [T,R,14,3])."""
    perm = L.tables()["swap_perm"][torch.as_tensor(aat)]
    sw = torch.gather(xl, -2, perm.expand(*xl.shape[:-1])[..., None].expand(*xl.shape))
    d1 = ((xl - ref) ** 2).sum((-1, -2)); d2 = ((sw - ref) ** 2).sum((-1, -2))
    return torch.where((d2 < d1)[..., None, None], sw, xl)


def residue_metrics(pred, pchi, x, tchi, dom, true_loc_c, dt):
    T, R = x.shape[:2]
    aat = dom.aatype
    m = dom.atom_mask.expand(T, R, 14)
    Rk, tk = kabsch(pred[:, :, :4].reshape(T, -1, 3), x[:, :, :4].reshape(T, -1, 3), m[..., :4].reshape(T, -1))
    P = (pred.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(pred.shape)
    scm = m.clone(); scm[..., :4] = 0
    d2 = ((P - x) ** 2).sum(-1); d2s = ((P - L._swapped(x, aat)) ** 2).sum(-1)
    sc_err = torch.minimum((d2 * scm).sum(-1), (d2s * scm).sum(-1)).sum(0).numpy()  # [R] summed over frames
    n_sc = scm[0].sum(-1).numpy()  # per frame
    err = L.chi_abs_err_deg(pchi, tchi, aat).numpy()  # [T,R,4]
    # local-frame RMSF of side-chain atoms
    Rp, tp = frames_from_backbone(pred[..., 0, :], pred[..., 1, :], pred[..., 2, :])
    pl = canon_local(to_local(pred, Rp, tp), true_loc_c[:1], aat)
    msf = lambda xl: (((xl - xl.mean(0)) ** 2).sum(-1).mean(0) * scm[0]).sum(-1) / scm[0].sum(-1).clamp(min=1)
    rmsf_p, rmsf_t = msf(pl).sqrt().numpy(), msf(true_loc_c).sqrt().numpy()
    pc, tc = ang(pchi.numpy()), ang(tchi.numpy())
    wp, wt = nearest_well(pc[..., 0], aat.numpy()), nearest_well(tc[..., 0], aat.numpy())
    cm = dom.chi_mask.numpy()
    hp, ht = hysteresis_states(pc, aat.numpy(), cm), hysteresis_states(tc, aat.numpy(), cm)
    _, rate_p = occ_and_rate(hp, dt)
    _, rate_t = occ_and_rate(ht, dt)
    return dict(sc_err=sc_err, n_sc=n_sc, chi_err=err.sum(0), chi1_ok=(err[..., 0] < 40).sum(0),
                state_ok=(wp == wt).sum(0), joint_ok=(hp == ht).sum(0), T=T,
                rmsf_p=rmsf_p, rmsf_t=rmsf_t, jsd=jsd(pops(wt), pops(wp)), rate_p=rate_p, rate_t=rate_t)


CLASSES = ("stable", "intermediate", "unstable", "not_stable", "all")


def summarise(rm, cls, cm):
    """pool residue metrics over a class mask -> sums (for pooling across systems) and means."""
    out = {}
    for c in CLASSES:
        sel = cls[c]
        if sel.sum() == 0:
            out[c] = dict(n_res=0); continue
        cmk = cm[sel]  # [n,4]
        s = dict(n_res=int(sel.sum()), T=rm["T"],
                 sc_err=float(rm["sc_err"][sel].sum()), n_sc=float(rm["n_sc"][sel].sum() * rm["T"]),
                 chi_err=[float((rm["chi_err"][sel, k] * cmk[:, k]).sum()) for k in range(4)],
                 n_chi=[float(cmk[:, k].sum() * rm["T"]) for k in range(4)],
                 chi1_ok=float(rm["chi1_ok"][sel].sum()), state_ok=float(rm["state_ok"][sel].sum()),
                 joint_ok=float(rm["joint_ok"][sel].sum()),
                 rmsf_p=float(rm["rmsf_p"][sel].sum()), rmsf_t=float(rm["rmsf_t"][sel].sum()),
                 jsd=float(rm["jsd"][sel].sum()), rate_p=float(rm["rate_p"][sel].sum()), rate_t=float(rm["rate_t"][sel].sum()))
        out[c] = s
    return out

