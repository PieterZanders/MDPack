# MDPack

A codec for compressing protein molecular dynamics trajectories, with a learned side-chain packer, residual entropy coding, and an optional backbone autoencoder.

The Python package name is `md_compress`, and the commands in this README are unchanged (`python -m md_compress.codec`, and so on).

## Requirements and installation

Run the commands below from the repository root. The default codec loads the shared model from `results_gpu/runs/e2e_S_prior/best.pt`.

```bash
pip install -r requirements.txt
```

The core dependencies are PyTorch, NumPy, SciPy and MDAnalysis, as listed in `requirements.txt`. Optional backends, also listed there as comments, are needed only for some `--method` choices:

* `zstandard` for `--method zstd`
* `h5py` for `--method h5md`, and `hdf5plugin` for `--h5md-compression sz3`
* `netCDF4` for `--method netcdf`

External tools, found on `PATH` or through an environment variable:

* GROMACS (`gmx`, or `MD_COMPRESS_GMX`) for `--method tng`. `--tng-precision` must be a power of ten.
* The MDCompress binary (`mdcompress`, or `MDCOMPRESS_BIN`) for `--method mdcompress`. The minimum resolution is 1000 fm.
* pyPcazip (`mdplus`) for `--method pcazip`.
* The compresstraj package, or `COMPRESSTRAJ_PATH`, for `--method compresstraj`.
* A cg2all installation (`MD_COMPRESS_CG2ALL_BIN` should point at its `bin` directory) for `--method cg2all`.
* PMC 1.0, the authors' tool for Dvořák, Maňák and Váša (J Mol Graph Model 2020), and a .NET runtime (8 or later), for `--method dvorak`. `bash md_compress/ext/get_pmc.sh PREFIX` downloads PMC (checksum verified) and a .NET 8 runtime; then set `MDPACK_PMC=PREFIX/pmc/bin/pmc.dll` and `MDPACK_DOTNET=PREFIX/dotnet/dotnet`.
* `mdz_file`, a file front end for MDZ (Zhao et al., ICDE 2022) built against SZ3, for `--method mdz`. `bash md_compress/ext/build_mdz.sh PREFIX` clones SZ3 at a pinned commit and builds it (needs git, cmake 3.19 or later, a C++17 compiler and libzstd); then set `MDPACK_MDZ=PREFIX/bin/mdz_file`. `./reproduce.sh --stage tools` runs both scripts.

The residual entropy coder is the C source `gpu/residual_entropy.c`. On first import, `gpu/residual_entropy.py` builds `gpu/_residual_entropy.so` with `gcc` if the library is missing or older than the source. You need `gcc` on `PATH`. The `.so` is not committed; you can also build it by importing the module:

```bash
python -c "from gpu import residual_entropy"
```

A GPU is not required for compress, decompress or the tests. Training (`train.py`, `gpu/pipeline.sh`, and the `train` stage of `reproduce.sh`) needs a GPU.

## Quick start

```bash
pip install -r requirements.txt
python -c "from gpu import residual_entropy"

python -m md_compress.codec compress   --top system.pdb --traj traj.xtc --out traj.mdz --bound 0.1
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.dcd
```

`decompress` writes `recon.json` next to `recon.dcd` unless you pass `--no-report`. Supply your own topology and trajectory; example MD files are not in this repository.

## Compress and decompress

The entry point is `md_compress/codec.py`. It writes a self describing `.mdz` file. The method, its options, the error bound and the topology are in the header, so `decompress` needs no method flags.

The default method is `new-coder`: no AI model; the coordinates (raw predictor) or their residuals against the trajectory mean structure after a per frame rigid fit (mean predictor) are quantised to the bound and coded by the context modelled coder in `gpu/residual_entropy.c`, with the bond sphere prediction described below. The encoder codes the file with both predictors and keeps the smaller one. Every atom is kept within the bound. Before this, the default was `hybrid`, which stays available with `--method hybrid` (and `method="hybrid"` in Python); existing hybrid files decode unchanged.

The `hybrid` method: the backbone (N, CA, C, O) and any extra atoms are coded against the trajectory mean structure after a per frame rigid fit; side chains are predicted by the shared pretrained packer from the decoded backbone (no rotamer codes, no per trajectory MLP); both residual streams go through the context modelled coder in `gpu/residual_entropy.c`. Every atom is kept within the bound.

```bash
python -m md_compress.codec compress   --top system.pdb --traj traj.xtc --out traj.mdz --bound 0.1
python -m md_compress.codec compress   --top system.pdb --traj traj.xtc --out traj.mdz --bound lossless
python -m md_compress.codec compress   --top system.pdb --traj traj.xtc --out traj.mdz --bound 0.5 --method hybrid --backbone ae
python -m md_compress.codec compress   --top system.pdb --traj traj.xtc --out traj.mdz --bound 0.1 --method hybrid --full-predictor
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.dcd
```

```python
from md_compress.codec import compress, decompress

blob = compress(coords, aatype, atom_mask, chain_index, chain_break, bound=0.1)  # new-coder; coords [T, R, 14, 3] in Angstrom
blob = compress(coords, aatype, atom_mask, chain_index, chain_break, bound="lossless", method="hybrid", backbone="ae",
                full_predictor=False, extra=extra_xyz)  # optional extra atoms [T, M, 3], coded with the backbone
r = decompress(blob)  # dict: coords, extra, aatype, atom_mask, chain_index, chain_break, bound, report (hybrid files also backbone, full_predictor)
blob, info = compress(..., return_info=True)  # info: size breakdown and options
```

| option | values | meaning |
|---|---|---|
| `bound` | a float in Angstrom, or `0` / `"lossless"` | maximum absolute error per coordinate. Lossless is exact on the 0.01 Angstrom grid (maximum error 0.005 Angstrom). |
| `backbone` | `"mean"` (default), `"ae"` | `"ae"` uses the shared backbone autoencoder (fp16 latents, lzma) instead of the mean structure. |
| `full_predictor` | `False` (default), `True` | per frame rotamer state codes and a per trajectory fine tuned MLP. |
| `angle_model` | `"mean"` (default), `"vonmises"`, `"empirical"`, `"gaussian"` | how the static per residue side chain torsion prior is estimated: circular mean (single von Mises maximum likelihood mean), mode of a 3 component von Mises mixture, histogram mode, or arithmetic mean (non circular). The prior is stored in the file, so decoding does not depend on it. CLI: `--angle-model`. |

Common flags for every method: `--stride`, `--threads`, `--bound`, and `--atoms protein` (default: atom14 protein heavy atoms plus OXT, required for the `hybrid` family and `cg2all`), `--atoms all`, or an MDAnalysis selection string. The selection is stored and reused on decompression.

The decoder recomputes packer (and autoencoder) predictions on a fixed single torch thread in both the encoder and the decoder, so `--threads` does not change the coordinates. Decode with the same code, model checkpoint and library build as the encoder; a different CPU or BLAS can change the last float bits of the prediction. Writing XTC or GRO can add up to 0.005 Angstrom of format rounding, and PDB up to 0.0005 Angstrom. Use DCD, TRR or the Python API when you need the bound to apply to the written file as well.

## Methods

`--method` selects the coder (default `new-coder`). Every method writes the same `.mdz` container. A flag that does not belong to the chosen method is rejected. `python -m md_compress.codec list-methods` prints the list and flags.

| method | what it stores | flags | error kind |
|---|---|---|---|
| `hybrid` | mean structure backbone, shared packer side chains, residuals, context modelled coder | `--bound B\|lossless` (default 0.1), `--backbone mean\|ae`, `--full-predictor`, `--coder ac\|lzma`, `--no-residuals`, `--angle-model mean\|vonmises\|empirical\|gaussian` | bounded (max B Angstrom); lossless uses the 0.01 Angstrom grid |
| `hybrid-ae` | as `hybrid`, with the backbone from the shared backbone autoencoder | same as `hybrid` | bounded or lossless, as `hybrid` |
| `model-only` | autoencoder latents, rotamer codes, per trajectory MLP, no residuals | `--angle-model` | no limit (RMSD and maximum error are reported) |
| `new-coder` (default) | raw coordinates or mean structure residuals, no AI model, bond sphere prediction in the coder | `--bound B\|lossless` (default 0.1), `--nc-predictor auto\|mean\|raw` (default auto: the smaller file), `--nc-coder ac\|lzma`, `--nc-sphere auto\|off\|sphere\|circle` (default auto), `--nc-choose full\|sub` (default full) | bounded or lossless |
| `xtc` | GROMACS XTC | `--xtc-precision P` (1/nm, default 1000) or `--bound B` (P = 5/B) | bounded (max 5/P Angstrom) |
| `tng` | GROMACS TNG, via `gmx trjconv` | `--tng-precision P` (power of ten, default 1000) or `--bound B` | bounded (max 5/P Angstrom) |
| `trr` | GROMACS TRR, uncompressed | `--trr-double` (float64) | exact |
| `dcd` | DCD, float32 | none | exact |
| `netcdf` | AMBER NetCDF, float32 | none | exact |
| `pdb` | multi model PDB text | `--pdb-precision 0..3` (default 3) or `--bound B`, `--pdb-gzip LEVEL` | bounded (max 0.5 × 10^(−P) Angstrom) |
| `h5md` | H5MD (HDF5), GROMACS layout (`/particles/system/position/value` in nm) | `--h5md-compression none\|gzip\|sz3`, `--h5md-level 1..9`, `--h5md-chunk FRAMES`, `--h5md-sz3-abs NM` or `--bound B` | exact for `none` and `gzip`; bounded for `sz3` |
| `mdcompress` | MDCompress v2 binary | `--mdc-res FM` or `--bound B\|lossless`, `--mdc-level L`, `--mdc-preset default\|archive\|trajectory\|frames`, `--mdc-batch N`, `--mdc-history H` | bounded (max FM/2 fm) or lossless |
| `pcazip` | pyPcazip 3 file, plus an optional per frame fit | `--pca-variance EV` (default 0.75) or `--pca-components N`, `--pca-residual-scale S` (per nm, default 200; 0 means PCA only), `--pca-eigenvector-scale`, `--pca-entropy lzma`, `--pca-frame lab\|fitted` | no limit (RMSD and maximum error are reported) |
| `compresstraj` | compresstraj autoencoder (decoder weights, latents, centres, scaler) | `--ct-compression C` (default 20) or `--ct-latent L`, `--ct-epochs E` (default 200), `--ct-layers 4096,1024`, `--ct-batch 128` | no limit (RMSD and maximum error are reported) |
| `cg2all` | coarse grained beads; all atoms rebuilt by cg2all | `--cg-model CalphaBasedModel\|BackboneModel\|ResidueBasedModel\|...`, `--cg-precision P` (1/nm, default 1000), `--cg-store xtc\|lzma`, `--cg-device cpu\|cuda` | no limit (RMSD and maximum error are reported) |
| `dvorak` | Dvořák, Maňák, Váša 2020, run by the authors' PMC 1.0: atoms visited along the bond graph, each predicted from a canonical molecule rotated about the bond to its predecessor (one quantised angle per atom and frame), residuals quantised in that local frame, Huffman coded; the bond list (pairs closer than the cutoff in frame 1) is stored | `--bound B` (default 0.1) or `--dv-qr Q` (residual step, Angstrom), `--dv-qa RAD` (angle step, default 0.1), `--dv-qc Q` (canonical molecule step, default 0.001), `--dv-cutoff A` (bond cutoff, default 2.1) | bounded (max B; with `--dv-qr Q`, max √3/2 × Q) |
| `mdz` | MDZ from SZ3 (`tools/mdz`), through `md_compress/ext/mdz_file.cpp`: per batch choice among VQ, VQT, MT, LR and TS predictors, then Huffman and zstd | `--bound B` (default 0.1), `--mdz-batch N` (frames per batch, default 10), `--mdz-predictor adp\|vq\|vqt\|mt\|lr\|ts` (default adp, adaptive), `--mdz-quantbins N` (default 1024) | bounded (max B) |
| `lzma`, `zstd`, `bz2`, `gzip` | quantised coordinates, time delta, zigzag, byte planes, then the general purpose compressor | `--bound B\|lossless\|exact`, `--level`, `--delta time\|none` | bounded, lossless, or exact (float32 bits) |

`model-only`, `pcazip`, `compresstraj` and `cg2all` ignore `--bound`. `dvorak` and `mdz` decode their own output before writing and refuse to write a file whose maximum error exceeds the bound; neither has a lossless mode. `xtc` and `tng` have no lossless or exact mode. `exact` is available for `lzma`, `zstd`, `bz2` and `gzip`. `trr`, `dcd`, `netcdf` and `h5md` without SZ3 store an exact copy of the float32 input. Default `--pca-frame lab` also stores the per frame rigid transform back to the input frame; `--pca-frame fitted` stores only pcazip's fitted frame. For `compresstraj`, the decoder weights are stored in the file because decompression needs them. Atoms that cg2all does not output (for example OXT) come back as NaN.

```bash
python -m md_compress.codec list-methods
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method xtc --xtc-precision 1000
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method tng --bound 0.05
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method mdcompress --bound 0.1 --mdc-level 4 --mdc-preset archive
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method new-coder --bound lossless --nc-predictor raw
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method h5md --h5md-compression sz3 --bound 0.1
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method pdb --pdb-gzip 9
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method pcazip --pca-variance 0.95 --pca-residual-scale 0
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method compresstraj --ct-compression 20 --ct-epochs 200
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method cg2all --cg-model CalphaBasedModel --cg-precision 1000
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method zstd --bound exact --level 22 --atoms all
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method dvorak --bound 0.1 --dv-cutoff 2.1
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --method mdz --bound 0.1 --mdz-batch 10 --mdz-predictor adp
python -m md_compress.codec compress --top sys.pdb --traj traj.xtc --out t.mdz --bound 0.1 --angle-model vonmises
python -m md_compress.codec decompress --in t.mdz --top sys.pdb --out recon.dcd
```

```python
from md_compress.codec import compress, compress_xyz, decompress

blob = compress(coords, aatype, atom_mask, chain_index, chain_break, method="mdcompress", bound=0.1, mdc_level=4)
blob, info = compress(coords, aatype, atom_mask, chain_index, chain_break, method="cg2all", return_info=True)
blob = compress_xyz(xyz, method="zstd", bound="exact", level=22)  # any atoms [T, N, 3]; decompress(blob)["xyz"]
```

### new-coder file format: coder blob versions 4 and 5

The new-coder payload holds the predictor data (mean predictor: mean structure, per frame rotation quaternions and translations, float32) and one coder blob. The first byte of the blob is its version:

* Version 4 (sphere off, unchanged): the per atom class linear predictor weights (16 classes, 4 int16 each), the reference atom offsets (lzma) and the three spatial neighbours per atom (lzma), then the arithmetic coded integers.
* Version 5 (bond sphere): the version 4 header, then one byte for the mode (1 sphere, 2 sphere plus circle), the bond plan per atom (offsets to the bonded neighbour B and to a second atom C, and whether C is taken from the current frame; int32, lzma), the mean bond lengths |x − B| (uint16, 0.001 Angstrom, lzma) and, for mode 2, the cosine of the mean angle at B (int16, 1e−4, lzma), then the arithmetic coded integers.

In mode 1, for every atom with a bonded neighbour B coded earlier in the same frame, the coordinate axis with the largest predicted offset from B is coded last and predicted from the sphere of the stored bond length around B, using the two axes already decoded. In mode 2 the middle axis is also predicted, from the circle given by the stored bond length and angle. Only addition, subtraction, multiplication, division and square root are used; when the square root argument is not positive, or a value is not finite, the atom falls back to the version 4 prediction, so a NaN never becomes an integer. The coder must be compiled with `-ffp-contract=off -fno-fast-math` (the build in `gpu/residual_entropy.py` does this); the portability check decoded identically with gcc and clang at -O0 to -O3 (also -march=native) and on aarch64 under qemu.

The encoder chooses the bonds from the original coordinates (for each atom the earlier atom whose distance varies least over the trajectory, among pairs with mean distance below 2.1 Angstrom and spread below 0.12 Angstrom) and stores them, so no topology is needed and the decoder does not depend on how they were found. With `--nc-sphere auto` it codes the file with the sphere off, then with sphere and sphere plus circle using the same predictor, and keeps the smallest blob. The mode is not written to the container options (the blob version carries it), so the container and the metadata trailer are the same for every mode and an auto file is never larger than with the sphere off. `--nc-choose sub` compares the modes on the first 128 frames only; it is faster but did not pick the same mode as the full comparison on every benchmark file, so `full` is the default.

## JSON report

`compress` stores a metadata trailer in every `.mdz` file: the method and options, the codec version, the input names and frame slicing, the system size, the size breakdown, the compress time, the resolution (error kind, limit and stored precision) and error statistics against the input. `decompress` writes that report as JSON next to the output trajectory (`recon.dcd` gives `recon.json`) unless you turn it off.

```bash
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.xtc
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.xtc --ref traj.xtc
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.xtc --report run1.json
python -m md_compress.codec decompress --in traj.mdz --top system.pdb --out recon.xtc --no-report
```

`--report` is on by default. `--report PATH` chooses the file. `--ref ORIGINAL` also recomputes errors against the original trajectory (same atoms and stride as at compression), both for the decoded coordinates and for the written file (including output format rounding). Files written before the metadata existed still decode; their report has `"metadata_present": false` and `"errors": null`.

```python
r = decompress(blob)                 # report=True (default): r["report"] is the report dict
r = decompress(blob, report=False)   # no report
r, rep = decompress(blob, return_info=True)
blob, info = compress(..., return_info=True)  # info["metadata"], info["metadata_bytes"], info["ratio_without_metadata"]
```

Fields in the JSON:

* Setup: `schema`, `writer`, `codec_version`, `container_version`, `method`, `options`.
* Input and system: `input` (topology, trajectory, atom selection, stride, start) and `system` (atom, residue and frame counts).
* Size: `size` with a per section `breakdown`, `core_bytes`, `metadata_bytes`, `total_bytes`, `ratio` and `ratio_without_metadata` (against float32). `ratio` includes the metadata.
* Timing: `compress_seconds`, and after decompress also `decompress_seconds`, `write_seconds`, `verify_decode_seconds`.
* Resolution: `resolution.kind` is `bounded`, `lossless`, `exact` or `no_limit`, plus `error_limit_angstrom`, `stored_precision` and `respects_bound`.
* Errors: for groups `all`, `backbone`, `side_chain` and `ca`: `rmsd_lab`, `rmsd_superposed_mean`, `abs_error` (mean, median, p99, max), `max_displacement`, `fraction_within_limit`, `per_frame_rmsd_lab` (mean, min, max) and `atoms_missing`. Absolute error is per Cartesian component. Also `bit_exact` and `measured_on`. Headline fields: `errors.rmsd_all`, `errors.rmsd_backbone` and `errors.rmsd_sidechain`, the per frame lab frame RMSD (no superposition) averaged over frames, in Angstrom; backbone = N, CA, C, O (plus OXT and any extra atoms coded with the backbone stream), side chain = the other protein heavy atoms (`errors.rmsd_is` states this in the file).
* Output: `output` (path, format, `extra_rounding_max_angstrom`). With `--ref`, `errors_vs_ref` holds `decoded` and `written_file` groups.

The following is a schema. Every value is a placeholder, not a measured result.

```json
{
  "schema": 1,
  "writer": "md_compress.codec",
  "codec_version": 1,
  "container_version": 1,
  "method": "<method>",
  "options": {"<flag>": "<value>"},
  "input": {
    "topology": "<topology.pdb>",
    "trajectory": "<trajectory.xtc>",
    "atoms": "protein",
    "stride": 1,
    "start": 0,
    "load_seconds": 0.0
  },
  "system": {"atoms": 0, "residues": 0, "frames": 0},
  "size": {
    "raw_float32_bytes": 0,
    "breakdown": {"<section>": 0},
    "core_bytes": 0,
    "metadata_bytes": 0,
    "total_bytes": 0,
    "ratio": 0.0,
    "ratio_without_metadata": 0.0
  },
  "compress_seconds": 0.0,
  "resolution": {
    "kind": "bounded",
    "error_limit_angstrom": 0.0,
    "stored_precision": {"grid": "<grid name>", "step_angstrom": 0.0},
    "respects_bound": true
  },
  "errors": {
    "measured_on": "the decoded file against the input, at compress time",
    "bit_exact": false,
    "percentiles_from_subsample": false,
    "tolerance_angstrom": 1e-05,
    "abs_error_is": "per Cartesian component, Angstrom",
    "groups": {
      "all": {
        "atoms": 0,
        "rmsd_lab": 0.0,
        "rmsd_superposed_mean": 0.0,
        "abs_error": {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0},
        "max_displacement": 0.0,
        "fraction_within_limit": 1.0,
        "per_frame_rmsd_lab": {"mean": 0.0, "min": 0.0, "max": 0.0},
        "atoms_missing": 0
      }
    }
  },
  "metadata_present": true,
  "decompress_seconds": 0.0,
  "write_seconds": 0.0,
  "output": {
    "path": "<recon.xtc>",
    "format": "xtc",
    "extra_rounding_max_angstrom": 0.005,
    "note": "XTC written at precision 1000 per nm (0.001 nm grid): adds up to 0.005 Angstrom per coordinate"
  },
  "decompressed_from": "<file.mdz>",
  "errors_vs_ref": {
    "reference": "<original.xtc>",
    "stride": 1,
    "start": 0,
    "decoded": {"groups": {"all": {}}},
    "written_file": {"groups": {"all": {}}}
  }
}
```

`errors_vs_ref` is present only when `--ref` is given. The `backbone`, `side_chain` and `ca` groups use the same fields as `all`.

## Running tests

The tests expect to be launched from the repository root and may need optional backends for the corresponding methods. `gcc` is required so the residual coder can be built.

```bash
python tests/test_codec.py
python tests/test_codec.py --quick
python tests/test_codec.py --methods-only
python tests/test_report.py
pytest tests/test_codec.py tests/test_report.py
```

`test_codec.py` checks compress and decompress round trips for the default codec and, with `--methods-only`, for every `--method`. `--quick` uses fewer frames. `test_report.py` checks the metadata and the JSON report, including `--report`, `--no-report` and older files in `tests/fixtures/`.

## Reproducing the report

`reproduce.sh` at the repository root runs the benchmark pipeline in stages. Use `--list` to print the stages, whether each needs a GPU, and the environments it expects. `--dry-run` prints the commands without running them.

```bash
./reproduce.sh --list
./reproduce.sh --stage report
./reproduce.sh --stage entropy --dry-run
./reproduce.sh --stage data,baselines
./reproduce.sh --stage tools,mdpack
./reproduce.sh --all
```

Stages, in the order used by `--all`:

| stage | GPU | what it needs |
|---|---|---|
| `data` | no | internet (HuggingFace mdCATH and MDposit) to download and export the trajectory sets |
| `train` | yes | a GPU machine; runs `gpu/pipeline.sh` |
| `baselines` | part | GROMACS, MDCompress and pyPcazip locally; compresstraj and cg2all are set up to run on Colab |
| `entropy` | no | the shared checkpoint in `results_gpu/runs/e2e_S_prior` and a built `gpu/_residual_entropy.so` |
| `scaling` | no | outputs of the `data` stage |
| `ablation` | no | the shared checkpoint and the built coder |
| `tools` | no | git, cmake, a C++17 compiler, libzstd and internet; builds `mdz_file` and fetches PMC 1.0 and a .NET runtime (`MDPACK_TOOLS`, default `~/.local/mdpack`) |
| `codec` | no | GROMACS, MDCompress, cg2all, pyPcazip, hdf5plugin and the `tools` stage, for the method tests |
| `report` | no | the result JSON produced by earlier stages (not tracked). It writes the Markdown, HTML and PDF report locally; those outputs are not tracked, and this stage does not change the README. |
| `mdpack` | no | the `tools` stage and the mdCATH domain 4m70B00 from the `data` stage (`MDPACK_MDCATH_LONG` for the atom14 conversion, `MDPACK_BENCH_DATA` for the PDB and reference frames). Runs every method on that one protein (`benchmark/scripts/bench_medium.py --system 4m70B00`) and builds the concise report `report_mdpack/MDPack_report.pdf` (`make_mdpack_report.py`; narrative text from `report_mdpack/report_text.json` if present). Outputs are not tracked. |

The script uses `BENCH_PY` (default: a conda env named `bench`) for the benchmark Python, and `SYS_PY` (default: `python3`) for the report build. Point those variables at interpreters that have the packages you need. Benchmark scripts skip rows already present in their output JSON; delete a result file to recompute it. Lines marked `[HAND]` in the script were done by hand in the original run; lines marked `[INFERRED]` were reconstructed from notes and logs.

## Repository layout

| path | contents |
|---|---|
| `md_compress/` | Python package: `codec.py` (CLI and API), `methods.py` (the `--method` backends), `report.py` (JSON report), `angles.py` (circular statistics for `--angle-model`), `ext/` (setup scripts and the MDZ file front end for `dvorak` and `mdz`), models, geometry and I/O |
| `gpu/` | Residual entropy coder (`residual_entropy.c`, `residual_entropy.py`), training and evaluation scripts, `pipeline.sh` |
| `tests/` | Codec and report tests, plus fixtures for older `.mdz` files |
| `tools/` | Helper scripts (figures, summaries, data fetch) |
| `results_gpu/runs/` | Pretrained checkpoints, including `e2e_S_prior/best.pt` used by the default codec, and each run's `config.json` |
| `results_gpu/frames/selected_domains.json` | The list of 16 mdCATH domain ids used by the benchmark scripts |
| `benchmark/` | Benchmark scripts and section builders. Generated reports, figures and result JSON are not tracked. |
| `reproduce.sh` | Stage runner for the benchmark pipeline |
| `requirements.txt` | Core Python dependencies |
| `example.py`, `train.py`, `infer.py`, `multi.py`, `validate_geometry.py` | Training, inference and geometry helpers (separate from `md_compress.codec`) |

```bash
python example.py --fast
python train.py --top system.pdb --traj traj.xtc --out ckpt.pt --codec residue --latent-dim 4
python infer.py compress   --ckpt ckpt.pt --top system.pdb --traj traj.xtc --out latents.npz
python infer.py decompress --ckpt ckpt.pt --top system.pdb --latents latents.npz --out recon.xtc --out-top recon.pdb
python validate_geometry.py
```
