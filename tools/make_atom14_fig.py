"""atom14 representation table straight from residue_constants: slot names, chi count, pi-periodic and symmetric atoms."""
import sys; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from md_compress import residue_constants as rc
out = sys.argv[1] if len(sys.argv) > 1 else "report/arch_figs/h_atom14.png"
types = rc.RESTYPES
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 16.6); ax.set_ylim(len(types) + 2.2, 0); ax.axis("off")
hdr = ["type"] + [str(i) for i in range(14)] + ["χ"]
xs = [0.5] + [1.4 + i * 1.0 for i in range(14)] + [15.6]
for x, h in zip(xs, hdr):
    ax.text(x, 0.5, h, ha="center", va="center", fontsize=9, weight="bold", color="#333")
sym = {r: {a for p in v for a in p} for r, v in rc.SYMMETRIC_SWAPS.items()}
for r, name in enumerate(types):
    y = r + 1.5
    ax.text(xs[0], y, name, ha="center", va="center", fontsize=9, weight="bold")
    names = rc.atom14_names(name)
    nchi = len(rc.chi_atoms(name))
    for s, a in enumerate(names):
        if not a: continue
        fc = "#F3F3F3" if s < 4 else ("#F5DDEA" if a in sym.get(name, ()) else "#DCE8F7")
        ax.add_patch(plt.Rectangle((xs[s + 1] - 0.47, y - 0.42), 0.94, 0.84, fc=fc, ec="#999", lw=0.5))
        ax.text(xs[s + 1], y, a, ha="center", va="center", fontsize=7.5)
    per = [k + 1 for k in rc.CHI_PI_PERIODIC.get(name, [])]
    ax.text(xs[-1], y, f"{nchi}" + (f" (χ{per[0]} π)" if per else ""), ha="center", va="center", fontsize=8)
for fc, lab, x in [("#F3F3F3", "backbone slots 0-3 (N, CA, C, O): from the codec", 0.2),
                   ("#DCE8F7", "side-chain slots: placed by NeRF from χ", 6.0),
                   ("#F5DDEA", "symmetric pair: loss takes min over swap", 10.6)]:
    ax.add_patch(plt.Rectangle((x, len(types) + 1.55), 0.4, 0.3, fc=fc, ec="#999", lw=0.5))
    ax.text(x + 0.55, len(types) + 1.7, lab, va="center", fontsize=8.5)
plt.tight_layout(); plt.savefig(out, dpi=160); print(out)
