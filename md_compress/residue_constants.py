"""
Residue constants: residue vocabulary, atom14-style per-residue layout, ideal
internal coordinates for NeRF side-chain building, chi-angle definitions,
symmetry information and van der Waals radii.

Geometry values are standard ideal values (Engh & Huber 1991/2001 style bond
lengths/angles; the same numbers used by most AMBER/CHARMM-like ideal residue
templates).  Every heavy side-chain atom is placed by NeRF from three
previously placed atoms (A, B, C) using

    bond  |C-D|,   angle  B-C-D,   torsion  A-B-C-D

The torsion is either a fixed value (rings, planar groups) or a chi angle plus
a constant offset (branches, e.g. VAL CG2 = chi1 + 122.6 deg).

Layout ("atom14"):  slot 0..3 = N, CA, C, O ; slots 4.. = side-chain heavy
atoms in the build order given below (TRP is the largest with 10 -> 14 slots).

If you prefer data-driven ideal geometry, `geometry.calibrate_from_coords`
(see validate_geometry.py) measures the same internal coordinates from a
structure; the oracle test in validate_geometry.py checks the table.
"""
from __future__ import annotations

import numpy as np

RESTYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
UNK = "UNK"
RESTYPE_ORDER = {r: i for i, r in enumerate(RESTYPES)}
NUM_RESTYPES = len(RESTYPES) + 1  # + UNK (index 20)
UNK_INDEX = len(RESTYPES)

# Force-field / PDB residue-name aliases -> canonical names
RESNAME_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS",
    "HIP": "HIS", "HISD": "HIS", "HISE": "HIS", "HISH": "HIS",
    "CYX": "CYS", "CYM": "CYS", "CYS2": "CYS", "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "LSN": "LYS", "MSE": "MET", "NALA": "ALA",
}
# Per-residue atom-name aliases (CHARMM/AMBER/GROMACS naming -> PDB naming)
ATOM_ALIASES = {
    "*": {"OT1": "O", "O1": "O", "OC1": "O"},
    "ILE": {"CD": "CD1"},
    "MET": {"SE": "SD"},
}
BACKBONE = ["N", "CA", "C", "O"]
ATOM14 = 14
MAX_CHI = 4

_CB = ("CB", "C", "N", "CA", 1.530, 110.5, -122.6)


# (name, A, B, C, bond[Å], angle B-C-D [deg], torsion A-B-C-D: float | "chiK" | ("chiK", offset))
SIDECHAIN_BUILD = {
    "ALA": [_CB],
    "ARG": [_CB,
            ("CG", "N", "CA", "CB", 1.520, 114.1, "chi1"),
            ("CD", "CA", "CB", "CG", 1.520, 111.5, "chi2"),
            ("NE", "CB", "CG", "CD", 1.460, 112.0, "chi3"),
            ("CZ", "CG", "CD", "NE", 1.330, 124.5, "chi4"),
            ("NH1", "CD", "NE", "CZ", 1.326, 120.6, 0.0),
            ("NH2", "CD", "NE", "CZ", 1.326, 119.6, 180.0)],
    "ASN": [_CB,
            ("CG", "N", "CA", "CB", 1.516, 112.6, "chi1"),
            ("OD1", "CA", "CB", "CG", 1.231, 120.8, "chi2"),
            ("ND2", "CA", "CB", "CG", 1.328, 116.4, ("chi2", 180.0))],
    "ASP": [_CB,
            ("CG", "N", "CA", "CB", 1.516, 112.6, "chi1"),
            ("OD1", "CA", "CB", "CG", 1.249, 118.4, "chi2"),
            ("OD2", "CA", "CB", "CG", 1.249, 118.4, ("chi2", 180.0))],
    "CYS": [_CB,
            ("SG", "N", "CA", "CB", 1.808, 114.0, "chi1")],
    "GLN": [_CB,
            ("CG", "N", "CA", "CB", 1.520, 114.1, "chi1"),
            ("CD", "CA", "CB", "CG", 1.516, 112.6, "chi2"),
            ("OE1", "CB", "CG", "CD", 1.231, 120.8, "chi3"),
            ("NE2", "CB", "CG", "CD", 1.328, 116.4, ("chi3", 180.0))],
    "GLU": [_CB,
            ("CG", "N", "CA", "CB", 1.520, 114.1, "chi1"),
            ("CD", "CA", "CB", "CG", 1.516, 112.6, "chi2"),
            ("OE1", "CB", "CG", "CD", 1.249, 118.4, "chi3"),
            ("OE2", "CB", "CG", "CD", 1.249, 118.4, ("chi3", 180.0))],
    "GLY": [],
    "HIS": [_CB,
            ("CG", "N", "CA", "CB", 1.497, 113.7, "chi1"),
            ("ND1", "CA", "CB", "CG", 1.378, 122.7, "chi2"),
            ("CD2", "CA", "CB", "CG", 1.354, 131.0, ("chi2", 180.0)),
            ("CE1", "CB", "CG", "ND1", 1.321, 109.0, 180.0),
            ("NE2", "CB", "CG", "CD2", 1.374, 107.0, 180.0)],
    "ILE": [_CB,
            ("CG1", "N", "CA", "CB", 1.530, 110.4, "chi1"),
            ("CG2", "N", "CA", "CB", 1.521, 110.5, ("chi1", -122.6)),
            ("CD1", "CA", "CB", "CG1", 1.513, 113.97, "chi2")],
    "LEU": [_CB,
            ("CG", "N", "CA", "CB", 1.530, 116.3, "chi1"),
            ("CD1", "CA", "CB", "CG", 1.521, 110.5, "chi2"),
            ("CD2", "CA", "CB", "CG", 1.521, 110.5, ("chi2", 122.6))],
    "LYS": [_CB,
            ("CG", "N", "CA", "CB", 1.520, 114.1, "chi1"),
            ("CD", "CA", "CB", "CG", 1.520, 111.5, "chi2"),
            ("CE", "CB", "CG", "CD", 1.520, 111.5, "chi3"),
            ("NZ", "CG", "CD", "CE", 1.489, 111.7, "chi4")],
    "MET": [_CB,
            ("CG", "N", "CA", "CB", 1.520, 114.1, "chi1"),
            ("SD", "CA", "CB", "CG", 1.807, 112.7, "chi2"),
            ("CE", "CB", "CG", "SD", 1.789, 100.9, "chi3")],
    "PHE": [_CB,
            ("CG", "N", "CA", "CB", 1.502, 113.8, "chi1"),
            ("CD1", "CA", "CB", "CG", 1.384, 120.7, "chi2"),
            ("CD2", "CA", "CB", "CG", 1.384, 120.7, ("chi2", 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.382, 120.7, 180.0),
            ("CE2", "CB", "CG", "CD2", 1.382, 120.7, 180.0),
            ("CZ", "CG", "CD1", "CE1", 1.382, 120.0, 0.0)],
    "PRO": [_CB,
            ("CG", "N", "CA", "CB", 1.495, 104.5, "chi1"),
            ("CD", "CA", "CB", "CG", 1.507, 105.5, "chi2")],
    "SER": [_CB,
            ("OG", "N", "CA", "CB", 1.417, 111.1, "chi1")],
    "THR": [_CB,
            ("OG1", "N", "CA", "CB", 1.433, 109.6, "chi1"),
            ("CG2", "N", "CA", "CB", 1.521, 111.5, ("chi1", -120.0))],
    "TRP": [_CB,
            ("CG", "N", "CA", "CB", 1.498, 113.6, "chi1"),
            ("CD1", "CA", "CB", "CG", 1.365, 126.9, "chi2"),
            ("CD2", "CA", "CB", "CG", 1.433, 126.6, ("chi2", 180.0)),
            ("NE1", "CB", "CG", "CD1", 1.374, 110.2, 180.0),
            ("CE2", "CB", "CG", "CD2", 1.409, 107.2, 180.0),
            ("CE3", "CB", "CG", "CD2", 1.398, 133.9, 0.0),
            ("CZ2", "CG", "CD2", "CE2", 1.394, 122.4, 180.0),
            ("CZ3", "CG", "CD2", "CE3", 1.382, 118.7, 180.0),
            ("CH2", "CD2", "CE2", "CZ2", 1.368, 117.5, 0.0)],
    "TYR": [_CB,
            ("CG", "N", "CA", "CB", 1.512, 113.8, "chi1"),
            ("CD1", "CA", "CB", "CG", 1.389, 120.8, "chi2"),
            ("CD2", "CA", "CB", "CG", 1.389, 120.8, ("chi2", 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.382, 121.2, 180.0),
            ("CE2", "CB", "CG", "CD2", 1.382, 121.2, 180.0),
            ("CZ", "CG", "CD1", "CE1", 1.378, 119.6, 0.0),
            ("OH", "CD1", "CE1", "CZ", 1.376, 119.9, 180.0)],
    "VAL": [_CB,
            ("CG1", "N", "CA", "CB", 1.524, 110.7, "chi1"),
            ("CG2", "N", "CA", "CB", 1.524, 110.4, ("chi1", 122.6))],
}

# chi angles that are pi-periodic because of symmetric terminal groups
CHI_PI_PERIODIC = {"ASP": [1], "GLU": [2], "PHE": [1], "TYR": [1]}  # 0-based chi index
# atom swaps that leave the residue chemically identical (for symmetry-aware losses)
SYMMETRIC_SWAPS = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ARG": [("NH1", "NH2")],
}
VDW_RADIUS = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def atom14_names(resname: str) -> list[str]:
    names = list(BACKBONE) + [a[0] for a in SIDECHAIN_BUILD.get(resname, [])]
    return names + [""] * (ATOM14 - len(names))


def _parse_torsion(spec):
    """-> (chi_index or -1, offset_or_fixed_deg)"""
    if isinstance(spec, str):
        return int(spec[3:]) - 1, 0.0
    if isinstance(spec, tuple):
        return int(spec[0][3:]) - 1, float(spec[1])
    return -1, float(spec)


def chi_atoms(resname: str) -> list[list[str]]:
    """Four atom names defining each chi (taken from the atom built with torsion == chiK, offset 0)."""
    out = {}
    for name, a, b, c, *_rest, tors in SIDECHAIN_BUILD.get(resname, []):
        k, off = _parse_torsion(tors)
        if k >= 0 and isinstance(tors, str):
            out[k] = [a, b, c, name]
    return [out[k] for k in sorted(out)]


def build_tables():
    """Numpy tables indexed by restype (21) and atom14 slot (14).

    Returns dict with:
      atom_mask      [21,14]  1 if the slot exists for this residue type
      ref_idx        [21,14,3] atom14 slot indices of (A,B,C) refs for NeRF (side-chain slots only)
      bond, angle    [21,14]   (Å, radians)
      chi_idx        [21,14]   chi index driving the torsion (-1 = fixed)
      torsion        [21,14]   fixed torsion or chi offset (radians)
      chi_mask       [21,4]    which chis exist
      chi_atom_idx   [21,4,4]  atom14 slots of the 4 atoms per chi
      chi_pi_periodic[21,4]
      swap_perm      [21,14]   permutation implementing symmetric swaps
      vdw            [21,14]   van der Waals radius per slot
      element        list of lists of element symbols
    """
    R = NUM_RESTYPES
    t = dict(
        atom_mask=np.zeros((R, ATOM14), np.float32),
        ref_idx=np.zeros((R, ATOM14, 3), np.int64),
        bond=np.zeros((R, ATOM14), np.float32),
        angle=np.zeros((R, ATOM14), np.float32),
        chi_idx=-np.ones((R, ATOM14), np.int64),
        torsion=np.zeros((R, ATOM14), np.float32),
        chi_mask=np.zeros((R, MAX_CHI), np.float32),
        chi_atom_idx=np.zeros((R, MAX_CHI, 4), np.int64),
        chi_pi_periodic=np.zeros((R, MAX_CHI), np.float32),
        swap_perm=np.tile(np.arange(ATOM14), (R, 1)),
        vdw=np.zeros((R, ATOM14), np.float32),
    )
    elements = []
    for ri in range(R):
        rn = RESTYPES[ri] if ri < len(RESTYPES) else "GLY"  # UNK -> backbone only
        names = atom14_names(rn)
        idx = {n: i for i, n in enumerate(names) if n}
        t["atom_mask"][ri, : len(idx)] = 1.0
        elements.append([n[0] if n else "" for n in names])
        for s, n in enumerate(names):
            if n:
                t["vdw"][ri, s] = VDW_RADIUS[n[0]]
        for s, (name, a, b, c, bond, ang, tors) in enumerate(SIDECHAIN_BUILD.get(rn, []), start=4):
            t["ref_idx"][ri, s] = [idx[a], idx[b], idx[c]]
            t["bond"][ri, s] = bond
            t["angle"][ri, s] = np.deg2rad(ang)
            k, off = _parse_torsion(tors)
            t["chi_idx"][ri, s] = k
            t["torsion"][ri, s] = np.deg2rad(off)
        for k, atoms in enumerate(chi_atoms(rn)):
            t["chi_mask"][ri, k] = 1.0
            t["chi_atom_idx"][ri, k] = [idx[a] for a in atoms]
        for k in CHI_PI_PERIODIC.get(rn, []):
            t["chi_pi_periodic"][ri, k] = 1.0
        for a, b in SYMMETRIC_SWAPS.get(rn, []):
            t["swap_perm"][ri, idx[a]], t["swap_perm"][ri, idx[b]] = idx[b], idx[a]
    t["element"] = elements
    return t


def restype_index(resname: str) -> int:
    rn = RESNAME_ALIASES.get(resname.upper(), resname.upper())
    return RESTYPE_ORDER.get(rn, UNK_INDEX)


def canonical_atom_name(resname: str, atom: str) -> str:
    rn = RESNAME_ALIASES.get(resname.upper(), resname.upper())
    atom = atom.upper()
    atom = ATOM_ALIASES["*"].get(atom, atom)
    return ATOM_ALIASES.get(rn, {}).get(atom, atom)
