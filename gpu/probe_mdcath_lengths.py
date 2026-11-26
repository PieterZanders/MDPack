"""Probe frame counts per temperature/replica for the mdCATH test domains (metadata only, no coordinates)."""
import json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor
import fsspec, h5py, requests
URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{}.h5"
OUT = "results_gpu/frames/mdcath_lengths.json"
os.makedirs("results_gpu/frames", exist_ok=True)
man = json.load(open("/workspace/md_compress_gpu_data/data_mdcath/manifest.json"))
test = sorted(d for d, v in man["domains"].items() if v["split"] == "test")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def resolve(url):
    for k in range(8):
        r = requests.head(url, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 307, 308): return r.headers["Location"]
        if r.status_code == 200: return url
        time.sleep(min(60, 5 * 2 ** k) * (0.5 + random.random()))
    raise RuntimeError(r.status_code)

def probe(d):
    with fsspec.open(resolve(URL.format(d)), block_size=2 ** 20).open() as fh:
        g = h5py.File(fh, "r")[d]
        return d, {t: {r: int(g[t][r]["coords"].shape[0]) for r in g[t]} for t in g if t.isdigit()}

with ThreadPoolExecutor(8) as ex:
    for d, v in ex.map(probe, [d for d in test if d not in res]):
        res[d] = v; json.dump(res, open(OUT, "w"), indent=1); print(d, {t: sum(x.values()) for t, x in v.items()}, flush=True)
