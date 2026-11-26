"""Rebuild /workspace/md_compress.zip (full) and /workspace/md_compress_code.zip (code + models + report, < 25 MB).
Run with system python3 (zip CLI is not installed on the box)."""
import fnmatch, os, shutil, zipfile
BASE = "/workspace"
EXCL = ["*/dev/*.npz", "*/dumps/*", "*__pycache__*", "*.so", "*.pyc"]
excl = lambda p: any(fnmatch.fnmatch(p, g) for g in EXCL)


def walk(d):
    for r, _, fs in os.walk(os.path.join(BASE, d)):
        for f in fs:
            p = os.path.relpath(os.path.join(r, f), BASE)
            if not excl(p): yield p


def rebuild(src, new_files, replace_dirs=(), refresh_all=False):
    tmp = src + ".new"; zin = zipfile.ZipFile(src); have = set(); n = 0
    new = set(new_files)
    for d in replace_dirs: new.update(walk(d))
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zin.infolist():
            nm = it.filename
            if nm in new or any(nm.startswith(d + "/") for d in replace_dirs): continue
            if refresh_all and os.path.isfile(os.path.join(BASE, nm)): zo.write(os.path.join(BASE, nm), nm)
            else: zo.writestr(it, zin.read(nm))
            n += 1
        for p in sorted(new):
            if os.path.isfile(os.path.join(BASE, p)): zo.write(os.path.join(BASE, p), p); n += 1
    shutil.move(tmp, src); z = zipfile.ZipFile(src); assert z.testzip() is None
    print(src, n, "entries", f"{os.path.getsize(src) / 1e6:.1f} MB")


CODE_NEW = ["md_compress/md_compress/codec.py", "md_compress/md_compress/methods.py", "md_compress/md_compress/report.py", "md_compress/reproduce.sh", "md_compress/tests/test_codec.py",
            "md_compress/tests/test_report.py", "md_compress/benchmark/entropy/results/report_tests.json"] + ["md_compress/tests/fixtures/" + f for f in __import__("os").listdir("/workspace/md_compress/tests/fixtures")] + [ "md_compress/README.md",
            "md_compress/benchmark/scripts/make_zips.py", "md_compress/gpu/residual_entropy.c", "md_compress/gpu/residual_entropy.py"] + \
           [f"md_compress/benchmark/entropy/results/{f}" for f in ("codec_tests.json", "codec_method_tests.json", "codec_summary.json")] + \
           [f"md_compress/benchmark/entropy/{f}" for f in ("codec_section.py", "codec_section.md", "STATE.md")] + \
           [f"md_compress/benchmark/benchmark_report.{e}" for e in ("md", "pdf")]
rebuild(f"{BASE}/md_compress.zip", ["md_compress/README.md", "md_compress/gpu/residual_entropy.c", "md_compress/gpu/residual_entropy.py",
                                    "md_compress/md_compress/codec.py", "md_compress/md_compress/methods.py", "md_compress/md_compress/report.py", "md_compress/reproduce.sh"], ["md_compress/benchmark", "md_compress/tests"])
rebuild(f"{BASE}/md_compress_code.zip", CODE_NEW, refresh_all=True)
