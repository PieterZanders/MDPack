"""Ray-traced PNGs (PyMOL open-source, headless): overlay, error-colored, pocket close-up."""
import json, sys
import pymol
pymol.finish_launching(["pymol", "-cq"])
from pymol import cmd

V = "report/viz"
info = json.load(open(f"{V}/viz_info.json"))


def setup():
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("stick_radius", 0.18)
    cmd.set("ray_shadows", 0)
    cmd.set("depth_cue", 1)
    cmd.set("specular", 0.2)


def load(name, f):
    cmd.load(f"{V}/{name}_true_f{f}.pdb", "true")
    cmd.load(f"{V}/{name}_recon_f{f}.pdb", "recon")
    cmd.hide("everything")


def render(path, w=1800, h=1350):
    cmd.ray(w, h)
    cmd.png(path, dpi=300)
    print("wrote", path)


for name in info:
    f = info[name]["frame"]
    sc = "not name N+C+O"  # side chain incl. CA for stick connectivity
    # 1. overlay
    setup(); load(name, f)
    cmd.show("cartoon", "true")
    cmd.color("grey80", "true")
    cmd.set("cartoon_transparency", 0.55, "true")
    cmd.show("sticks", f"true and {sc} and not resn GLY")
    cmd.show("sticks", f"recon and {sc} and not resn GLY")
    cmd.color("grey40", "true and elem C")
    cmd.color("orange", "recon and elem C")
    cmd.util.cnc("all")
    cmd.color("grey40", "true and elem C")
    cmd.color("orange", "recon and elem C")
    cmd.orient("true")
    render(f"{V}/{name}_overlay.png")
    # 2. error-colored reconstruction
    setup(); load(name, f)
    cmd.show("cartoon", "recon")
    cmd.show("sticks", f"recon and {sc} and not resn GLY")
    cmd.spectrum("b", "blue_white_red", "recon", minimum=0, maximum=2.5)
    cmd.set("cartoon_transparency", 0.3, "recon")
    cmd.orient("recon")
    render(f"{V}/{name}_error.png")
    # 3. pocket close-up around the most packed aromatic/large residue in the core
    setup(); load(name, f)
    cmd.select("big", "true and name CA and resn PHE+TYR+TRP+LEU+ILE+MET")
    best, bestn = None, -1
    for at in cmd.get_model("big").atom:
        n = cmd.count_atoms(f"true and name CA within 9 of (true and resi {at.resi} and name CA)")
        if n > bestn:
            best, bestn = at.resi, n
    pocket = f"byres (true within 6.5 of (true and resi {best} and {sc}))"
    cmd.select("pk", f"resi {'+'.join(sorted({a.resi for a in cmd.get_model(pocket).atom}, key=int))}")
    cmd.show("cartoon", "true")
    cmd.color("grey90", "true")
    cmd.set("cartoon_transparency", 0.7, "true")
    cmd.show("sticks", f"pk and {sc} and not resn GLY")
    cmd.util.cnc("all")
    cmd.color("grey40", "true and elem C")
    cmd.color("orange", "recon and elem C")
    cmd.set("label_size", 14)
    cmd.set("label_position", (0, 1.5, 2))
    cmd.set("label_color", "black")
    cmd.label(f"true and pk and name CB", '"%s%s" % (resn, resi)')
    cmd.orient("pk and true")
    cmd.zoom("pk and true", 1.0)
    render(f"{V}/{name}_pocket.png")
    info[name]["pocket_center"] = best
    info[name]["pocket_residues"] = [f"{a.resn}{a.resi}" for a in cmd.get_model("pk and true and name CA").atom]
json.dump(info, open(f"{V}/viz_info.json", "w"), indent=1)
