#!/usr/bin/env python
"""Architecture / data-flow / experimental-design diagrams (graphviz). All dims are taken from the code
(md_compress/*.py, gpu/common.py VARIANTS, gpu/train_gpu.py, multi.py, example.py). Output: report/arch_figs/*.png"""
import os, subprocess, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "report/arch_figs"
os.makedirs(OUT, exist_ok=True)
STY = {
    "data": ("#F3F3F3", "#7A7A7A"), "learn": ("#DCE8F7", "#2F64B0"), "geo": ("#E2F1DF", "#3F7F38"),
    "frame": ("#FCE6C9", "#C0741A"), "prot": ("#F5DDEA", "#9C3F72"), "loss": ("#FADCD9", "#B03A30"),
    "note": ("#FFFFFF", "#AAAAAA"), "base": ("#EDEDED", "#555555"),
}
HEAD = '''digraph G {
 graph [fontname="DejaVu Sans", fontsize=11, bgcolor="white", pad=0.25, nodesep=0.28, ranksep=0.32, %s];
 node  [fontname="DejaVu Sans", fontsize=10, shape=box, style="rounded,filled", penwidth=1.3, margin="0.12,0.06"];
 edge  [fontname="DejaVu Sans", fontsize=9, color="#555555", arrowsize=0.7, penwidth=1.1];
'''


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def N(name, title, lines=(), kind="data", extra=""):
    f, c = STY[kind]
    body = "".join(f'<BR/><FONT POINT-SIZE="8.5" COLOR="#333333">{esc(l)}</FONT>' for l in lines)
    return f' {name} [label=<<B>{esc(title)}</B>{body}>, fillcolor="{f}", color="{c}" {extra}];\n'


def legend(kinds):
    names = dict(note="annotation", data="tensor / data", learn="learned module", geo="parameter-free geometry", frame="stored per frame",
                 prot="stored once per protein", loss="loss / metric", base="baseline / reference")
    cells = "".join(f'<TD BGCOLOR="{STY[k][0]}" COLOR="{STY[k][1]}" BORDER="1" STYLE="rounded"><FONT POINT-SIZE="8.5">{names[k]}</FONT></TD>'
                    for k in kinds)
    return (f' legend [shape=plaintext, style="", label=<<TABLE BORDER="0" CELLSPACING="6" CELLPADDING="4"><TR>{cells}</TR></TABLE>>];\n')


def render(fname, body, gattr='rankdir=TB'):
    src = HEAD % gattr + body + "}\n"
    open(f"{OUT}/{fname}.dot", "w").write(src)
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", f"{OUT}/{fname}.dot", "-o", f"{OUT}/{fname}.png"], check=True)
    print("wrote", fname)


# ---------------------------------------------------------------- (a) end-to-end pipeline
b = ""
b += N("traj", "MD trajectory", ["any MDAnalysis topology + trajectory", "or mdCATH h5 coords"], "data")
b += N("load", "load_trajectory", ["protein, standard residues, no H", "atom14 coords [T,R,14,3] + masks"], "geo")
b += N("bb", "backbone N,CA,C,O", ["[T,R,4,3]"], "data")
b += N("align", "Kabsch on CA to reference", ["aligned bb [T,R,4,3]", "rotation [T,3,3], translation [T,3]"], "geo")
b += N("enc", "Encoder (ResidueLatentCodec)", ["input: (bb − ref)/scale [T,12,R] + aa emb 8 + ref 12", "3 × Conv1d k=5, 64 ch, SiLU → Conv1d 1×1", "→ z [T,R,4]"], "learn")
b += N("store", "Stored per frame", ["latents float16 [R,4] = 8R bytes", "quaternion 4 + translation 3 (float32, 28 B)"], "frame")
b += N("pstore", "Stored once per protein", ["topology, ref backbone [R,4,3], scale", "χ prior [R,4,2] (circular mean)"], "prot")
b += N("dec", "Decoder", ["input: z [T,4,R] + aa emb 8 + ref 12", "3 × Conv1d k=5, 64 ch → Conv1d 1×1 → 12", "bb̂ = ref + Δ·scale  [T,R,4,3]"], "learn")
b += N("map", "MappingNet (residue graph GNN)", ["CA−CA < 10 Å edges, 3 MP layers (S)", "node 48 / edge 32 → context [T,R,32]"], "learn")
b += N("sc", "SideChainModel (shared by 20 types)", ["MLP trunk d=80, 2 residual blocks", "head → χ (sin,cos) [T,R,4,2]"], "learn")
b += N("nerf", "SideChainBuilder (NeRF)", ["ideal bonds/angles per type, no params", "→ atom14 [T,R,14,3]"], "geo")
b += N("refine", "refinement: 2 extra passes", ["previous χ, side-chain centroids,", "min side-chain distances fed back"], "note", 'style="rounded,dashed", fontcolor="#333333"')
b += N("unalign", "undo rigid transform", ["x = (x_al − t) R", "+ OXT placed from N,CA,C,O"], "geo")
b += N("write", "TrajectoryData.write", ["heavy-atom PDB / DCD / XTC", "with the original topology"], "data")
b += "traj->load->bb->align->enc->store; store->dec [label=\" decompress\"]; pstore->enc [style=dashed]; pstore->dec [style=dashed];\n"
b += "dec->map->sc->nerf->unalign->write; nerf->refine [style=dashed]; refine->map [style=dashed];\n"
b += "pstore->sc [style=dashed, label=\" χ prior\"]; align->store [label=\" R,t\", style=dotted];\n"
b += "{rank=same; enc; pstore} {rank=same; store; dec}\n"
b += legend(["data", "learn", "geo", "frame", "prot"])
render("a_pipeline", b)

# ---------------------------------------------------------------- (b) preprocessing and batching
b = ""
b += ' subgraph cluster_src { label="Sources"; color="#CCCCCC"; style=rounded;\n'
b += N("adk", "AdK (MDAnalysisTests)", ["adk.psf + adk_dims.dcd, 214 res, 98 frames"], "data")
b += N("atlas", "ATLAS (7 systems)", ["PDB + R1 xtc, stride 100 → 101 frames"], "data")
b += N("mdc", "mdCATH h5 (HuggingFace)", ["HTTP range reads (h5py + fsspec)", "320 K, replicas 0,1, every 2nd frame (2 ns)"], "data")
b += " }\n"
b += N("sel", "MDAnalysis selection", ["select 'protein'; drop residues not in the 20 types", "(force-field aliases HSD, CYX, ... mapped)", "drop hydrogens (by name / element); OXT kept apart"], "geo")
b += N("a14", "atom14 layout", ["coords [T,R,14,3] float32 Å, slots 0-3 = N,CA,C,O", "atom_mask [R,14], aatype [R] (21 types)", "slot_to_atom map for writing back"], "data")
b += N("brk", "chains and breaks", ["chain_index from segid", "chain_break: new chain, residue gap,", "or C(i−1)−N(i) > 2 Å in frame 0"], "geo")
b += N("npz", "per-domain npz (GPU)", ["coords, replica, masks, aatype, chain info, n_heavy, seq, pdb", "train coords packed in one memmap [ΣT·R,14,3]"], "data")
b += N("chi", "χ extraction", ["chis_from_atom14 → (sin,cos) [T,R,4,2], chi_mask [R,4]", "π-periodic χ (ASP χ2, GLU χ3, PHE/TYR χ2) canonicalised", "symmetric atom swaps (ASP, GLU, PHE, TYR, ARG) for losses"], "geo")
b += N("buf", "per-protein stored data", ["iterative mean: align on CA to frame 0, average, 3 iterations", "ref [R,4,3], scale = std(aligned − ref) ≥ 0.1 Å", "χ prior = circular mean of canonical χ [R,4,2]"], "prot")
b += N("split", "splits", ["CPU: every 5th frame held out (interpolation)", "folds: 4 × 2 systems held out", "GPU: one domain per CATH superfamily → 300 / 30 / 50"], "base")
b += N("batch", "GPU batch (one protein per step)", ["pick a training domain uniformly, bind its buffers", "draw B frames without replacement (S 32, L 16, e2e 16)", "x [B,R,14,3]; aatype [R]; chain_index [R]; chain_break [R]"], "data")
b += N("frames", "per step, in the model", ["codec: Kabsch align bb → [B,R,4,3]", "packer: residue frames N,CA,C (Gram-Schmidt)", "local coords x_loc = Rᵀ(x − CA): rotation/translation invariant"], "geo")
b += "adk->sel; atlas->sel; mdc->sel; sel->a14; sel->brk; a14->npz; brk->npz; npz->chi; npz->buf; chi->buf; npz->split; split->batch; buf->batch [style=dashed, label=\" bind()\"]; batch->frames;\n"
b += legend(["data", "geo", "prot", "base"])
render("b_preprocessing", b)

# ---------------------------------------------------------------- (c) side-chain model
b = ""
b += N("bbin", "backbone [B,R,4,3]", [], "data")
b += N("loc", "local frame", ["e1 = CA→C, e2 ⟂ in N-CA-C plane, e3 = e1 × e2", "N, C, O in own frame → 9"], "geo")
b += N("aa", "residue type embedding", ["Embedding(21, 16)  (L: 32)"], "learn")
b += N("ctx", "context from MappingNet", ["[B,R,32]  (L: 64)"], "data")
b += N("prior", "χ prior (sin,cos)", ["[R,4,2] → 8, zeros if no prior"], "prot")
b += N("cat", "concat", ["9 + 16 + 32 + 8 = 65  (L: 113)"], "data")
b += N("trunk", "trunk", ["Linear 65 → 80", "x = LN(x + MLP(SiLU x))  × 2 blocks", "MLP = Linear 80→80, SiLU, Linear 80→80"], "learn")
b += N("head", "SideChainHead (interface)", ["forward(h, chi_mask, chi_true, chi_ref) → chi_sc, nll", "sample(h, chi_mask, n) → [n,...,4,2]"], "learn")
b += N("det", "DeterministicChiHead (used)", ["Linear 80 → 8 = 4 × (sin,cos)", "normalise to unit circle → rotation δ", "χ = δ ∘ χ_prior (angle addition)", "init: weights × 0.1, bias = (0,1) = no rotation"], "learn")
b += N("vm", "VonMisesMixtureHead (option)", ["autoregressive χ1..χ4, K = 3 von Mises", "exact NLL, samples rotamers (+6.1k params)"], "learn", 'style="rounded,filled,dashed"')
b += N("flow", "torus flow (drop-in)", ["subclass SideChainHead, add to HEADS", "nothing else changes"], "note", 'style="rounded,dashed"')
b += N("mask", "χ mask per type", ["[R,4]: GLY/ALA 0, SER 1, ..., ARG/LYS 4"], "geo")
b += N("nerf", "SideChainBuilder (NeRF)", ["for slots 4..13: place atom from 3 parents,", "ideal bond, angle; torsion = χ_k + offset or fixed", "→ atom14 [B,R,14,3], mask [R,14]"], "geo")
b += "bbin->loc->cat; aa->cat; ctx->cat; prior->cat; cat->trunk->head; head->det; head->vm [style=dashed]; head->flow [style=dashed];\n"
b += "det->nerf [label=\" χ (sin,cos) [B,R,4,2]\"]; mask->det [style=dashed]; prior->det [style=dashed]; bbin->nerf [style=dotted];\n"
b += "{rank=same; aa; ctx; prior; loc}\n"
b += legend(["data", "learn", "geo", "prot"])
render("c_sidechain_model", b)

# ---------------------------------------------------------------- (d) mapping GNN
b = ""
b += ' subgraph cluster_node { label="Node features (53)"; color="#CCCCCC"; style=rounded;\n'
b += N("n1", "type embedding 16", [], "learn") + N("n2", "local N, C, O 9", [], "geo") + N("n3", "φ, ψ, ω (sin,cos) 6 + valid mask 3", [], "geo")
b += N("n4", "pass embedding 8", ["Embedding(4, 8)"], "learn") + N("n5", "previous side chain 11", ["χ (sin,cos) 8 + centroid in local frame 3", "zeros in pass 0"], "data")
b += " }\n"
b += ' subgraph cluster_edge { label="Edge features (55), edges j→i with CA−CA < 10 Å"; color="#CCCCCC"; style=rounded;\n'
b += N("e1", "RBF(d_CA) 16", ["0-10 Å"], "geo") + N("e2", "direction of j in i's frame 3", [], "geo") + N("e3", "relative rotation R_iᵀR_j 9", [], "geo")
b += N("e4", "sequence separation 8", ["Embedding(34, 8): clip ±16, other chain"], "learn")
b += N("e5", "previous side chain 19", ["RBF centroid distance 8 (0-16 Å)", "RBF min sc(i)-atom(j) distance 8 (0-8 Å)", "centroid direction 3; zeros in pass 0"], "data")
b += " }\n"
b += N("nin", "Linear 53 → 48  (L: 128)", [], "learn")
b += N("ein", "Linear 55 → 32  (L: 64)", [], "learn")
b += N("mp", "MPLayer × 3  (L: × 4)", ["m_ij = MLP([h_i, h_j, e_ij])  (2·48+32 → 48 → 48)", "a_i = mean_j m_ij", "h_i ← LN(h_i + MLP([h_i, a_i]))"], "learn")
b += N("out", "Linear 48 → 32  (L: 128 → 64)", ["context [B,R,32]"], "learn")
b += N("scm", "SideChainModel + NeRF", ["χ, atom14 of this pass"], "learn")
b += N("loop", "refinement loop: pass p = 0, 1, 2", ["same weights in every pass", "outputs detached before feeding back", "loss: last pass × 1.0, earlier × 0.5"], "note", 'style="rounded,dashed"')
b += "n1->nin; n2->nin; n3->nin; n4->nin; n5->nin; e1->ein; e2->ein; e3->ein; e4->ein; e5->ein; nin->mp; ein->mp; mp->out->scm; scm->loop [style=dashed];\n"
b += "loop->n5 [style=dashed, constraint=false, label=\" χ, centroids\"]; loop->e5 [style=dashed, constraint=false, label=\" centroids, min distances\"];\n"
b += legend(["data", "learn", "geo", "note"])
render("d_mapping_gnn", b, 'rankdir=LR, newrank=true')

# ---------------------------------------------------------------- (e) codecs
b = ""
b += N("al", "aligned backbone [T,R,4,3]", ["Kabsch on CA to the per-protein reference", "rigid part: quaternion 4 + translation 3 per frame"], "data")
b += ' subgraph cluster_res { label="ResidueLatentCodec (used everywhere on GPU)"; color="#2F64B0"; style=rounded;\n'
b += N("r1", "Δ = (bb − ref)/scale → [T,12,R]", ["+ aa embedding 8 + ref coords/10 12 = 32 channels"], "data")
b += N("r2", "encoder", ["Conv1d 32→64 k5, SiLU, Conv1d 64→64 k5 ×2, Conv1d 64→4 1×1", "receptive field ±6 residues"], "learn")
b += N("r3", "z [T,R,4] → float16", ["8 bytes per residue per frame", "~11× vs 12 B per heavy atom (about 7.7 atoms/res)"], "frame")
b += N("r4", "decoder", ["[4 + 8 + 12 = 24, R] → same conv stack → 12", "bb̂ = ref + Δ̂ · scale"], "learn")
b += N("r5", "weights independent of R", ["101,432 params, shared across proteins"], "note")
b += " }\n"
b += ' subgraph cluster_frm { label="FrameLatentCodec (single-protein option)"; color="#888888"; style="rounded,dashed";\n'
b += N("f1", "flatten Δ → [T, 12R]", [], "data")
b += N("f2", "MLP 12R → 256 → 32", ["nonlinear PCA"], "learn")
b += N("f3", "z [T,32] → float16", ["64 B per frame regardless of R"], "frame")
b += N("f4", "MLP 32 → 256 → 12R", ["1,334,312 params for AdK (R = 214)", "tied to one topology"], "learn")
b += " }\n"
b += "al->r1->r2->r3->r4; r4->r5 [style=invis]; al->f1->f2->f3->f4;\n"
b += legend(["data", "learn", "frame"])
render("e_codecs", b)

# ---------------------------------------------------------------- (f) training stages
b = ""
b += N("s0", "buffers", ["per protein: ref, scale, χ prior", "from its whole trajectory"], "prot")
b += N("s1", "Stage 1: codec", ["Kabsch-aligned bb → z → bb̂", "loss = MSE(bb̂, bb) + 1e-4·mean(z²)", "AdamW lr 2e-3, no weight decay"], "learn")
b += N("s2", "Stage 2: packer on TRUE backbone", ["mapping net + side-chain model (teacher forcing)", "AdamW lr 1e-3 (GPU) / 2e-3 (CPU), wd 1e-4", "S and L, with and without χ prior"], "learn")
b += N("s3", "Stage 3: end-to-end", ["z → bb̂ → packer, all weights", "init from codec + packer, lr 3e-4 (GPU) / 5e-4 (CPU)"], "learn")
b += N("lp", "packer loss (per pass p)", ["1.0 · side-chain MSE in each residue's own frame (symmetry min)", "1.0 · χ loss 1 − cos Δχ (|cos| for π-periodic)", "0.02 · (‖raw χ vector‖ − 1)²", "pass weights: last 1.0, earlier 0.5"], "loss")
b += N("lc", "soft clash (last pass)", ["0.1 · Σ relu(r_a + r_b − 1.5 Å − d)² / (B·R)", "pairs CA−CA < 14 Å, not sequence neighbours,", "≥ 1 side-chain atom, CYS SG-SG allowed"], "loss")
b += N("le", "end-to-end extra", ["1.0 · backbone MSE (aligned frame)", "0.5 · side-chain MSE in the global aligned frame"], "loss")
b += N("sch", "schedule (GPU)", ["cosine over wall-clock budget, 200-step warm-up, floor 2%", "grad clip 1.0, val every 180 s on 30 val domains", "best.pt by val score, last.pt for resume"], "note")
b += "s0->s1; s0->s2; s1->s3; s2->s3; lp->s2 [style=dashed]; lc->s2 [style=dashed]; lp->s3 [style=dashed]; le->s3 [style=dashed]; lc->s3 [style=dashed]; s3->sch [style=invis];\n"
b += "{rank=same; s1; s2}\n"
b += legend(["learn", "loss", "prot"])
render("f_training", b)

# ---------------------------------------------------------------- (g) experimental design
b = ""
def col(k, title, setup, test, comp, q):
    out = f' subgraph cluster_{k} {{ label="{title}"; color="#888888"; style=rounded; fontsize=11;\n'
    out += N(f"{k}s", "training data", setup, "data") + N(f"{k}t", "test data", test, "data")
    out += N(f"{k}c", "comparisons", comp, "learn") + N(f"{k}q", "question answered", q, "note")
    out += f" {k}s->{k}t->{k}c->{k}q;\n }}\n"
    return out
b += col("e1", "1. One protein, CPU (example.py)",
         ["AdK, 214 res, 98 frames", "4 of every 5 frames (80%)"], ["every 5th frame (interleaved)"],
         ["refinement passes 0 / 1 / 2", "χ prior vs none; deterministic vs von Mises", "per-residue vs per-frame codec"],
         ["does the pipeline work end to end", "on one topology (interpolation)?"])
b += col("e2", "2. Eight proteins, CPU (multi.py)",
         ["(a) one model per protein, 80% of frames", "(b) shared model on 6 proteins,", "40 frames per protein per epoch"],
         ["(a) every 5th frame of the same protein", "(b) the 2 held-out proteins, 4 folds"],
         ["per-protein vs shared model", "χ prior vs none", "vs mean rotamers and floor"],
         ["does one shared model transfer", "to proteins it has never seen?"])
b += col("e3", "3. mdCATH, GPU (gpu/)",
         ["300 domains = 300 superfamilies", "139,670 frames; val 30 superfamilies"],
         ["50 domains from 50 other superfamilies", "external: AdK + 7 ATLAS (other force fields)"],
         ["S (86k) vs L (745k)", "staged vs end-to-end; prior vs none", "true backbone vs decoded backbone"],
         ["does more data close the gap; is the", "model capacity or data limited; are", "dynamics (RMSF, PCA) preserved?"])
b += N("bl", "Baselines, all on the TRUE backbone through the same NeRF builder", ["ideal-geometry floor: true χ (error from ideal bonds and angles only)", "mean rotamers: the stored χ prior alone (cost: 32 B per residue, once)", "one rotamer per residue type: training-set mode, no per-protein data", "MD clash: clash score of the real MD frames"], "base")
b += N("mt", "Metrics", ["bb RMSD and symmetry-aware sc RMSD after superposing the backbone", "χ1 to χ4 MAE (π-periodic aware), χ1 within 40°, clashes per 1000 atoms", "compression: float32 heavy atoms per frame / stored bytes per frame", "dynamics: RMSD(t), CA and sc RMSF (Pearson r), PCA, RMSIP top 10"], "loss")
b += "e1q->bl [style=invis]; e2q->bl [style=invis]; e3q->bl [style=invis]; bl->mt [style=invis];\n"
b += "{rank=same; e1s; e2s; e3s}\n"
b += legend(["data", "learn", "note", "base", "loss"])
render("g_experimental_design", b, 'rankdir=TB, newrank=true')
