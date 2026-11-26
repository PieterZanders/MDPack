"""Parameter counting and memory estimation (measured saved-activation bytes + analytic scaling)."""
from __future__ import annotations

import torch


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_table(model) -> dict:
    return dict(
        codec=count_params(model.codec),
        mapping_net=count_params(model.packer.mapping),
        sidechain_model=count_params(model.packer.sidechain),
        sidechain_stack_total=count_params(model.packer),
        total=count_params(model),
    )


class SavedTensorMeter:
    """Counts bytes of tensors autograd saves for backward (= training activation memory)."""

    def __init__(self):
        self.bytes = 0
        self._seen = set()

    def pack(self, t):
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key not in self._seen:
            self._seen.add(key)
            self.bytes += t.numel() * t.element_size()
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        self._ctx.__exit__(*a)


def measure_training_activations(model, t, frame_idx, refine=None):
    """Saved-for-backward bytes of one end-to-end training forward pass on the given frames."""
    x = t["coords"][frame_idx]
    bb_al, _, _ = model.codec.align(x[:, :, :4])
    with SavedTensorMeter() as m:
        z = model.codec.encode(bb_al, t["aatype"])
        bb, outs = model.decode_full(z, t["aatype"], t["chain_index"], t["chain_break"], n_refine=refine)
        loss = sum(o["atom14"].pow(2).mean() for o in outs)
    del loss
    return m.bytes


def analytic_estimate(n_res=300, k=30, d_node=48, d_edge=32, n_layers=3, n_passes=3, d_sc=80,
                      batch=1, latent_dim=4, n_heavy_per_res=7.8, bytes_per=4):
    """Rough order-of-magnitude memory model (float32).

    Per pass the dominant terms are edge tensors: E = n_res*k edges, each layer saves the message
    MLP input (2*d_node+d_edge) and hidden/output (2*d_node) activations -> ~(4*d_node+d_edge) floats
    per edge per layer; refinement passes add a 14x14 distance block per edge (196 floats).
    """
    E = n_res * k
    per_pass_edge = E * (n_layers * (4 * d_node + d_edge) + 60)
    per_pass_refine = E * 196 * 2
    per_pass_node = n_res * (n_layers * 6 * d_node + 6 * d_sc + 14 * 3 * 10)
    train = batch * n_passes * (per_pass_edge + per_pass_refine + per_pass_node) * bytes_per
    # inference keeps only one layer's activations alive at a time (no_grad)
    infer = batch * (E * ((2 * d_node + d_edge) * 2 + 196) + n_res * 14 * 3 * 4) * bytes_per
    stored_latent = n_res * latent_dim * 2 + 7 * 4  # float16 latents + fp32 rigid
    raw = n_res * n_heavy_per_res * 3 * 4
    return dict(n_res=n_res, k=k, edges=E, train_activation_MB=train / 1e6, infer_activation_MB=infer / 1e6,
                stored_bytes_per_frame=stored_latent, raw_fp32_bytes_per_frame=raw,
                ratio=raw / stored_latent)
