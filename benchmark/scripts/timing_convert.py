#!/usr/bin/env python
"""results/timingrun_<x>.json (full records from the clean timing pass) -> results/timing_<x>.json {sys/T/method: {encode_s, decode_s}}."""
import glob, json, os
R = "/workspace/md_compress/benchmark/results"
for f in glob.glob(f"{R}/timingrun_*.json"):
    d = json.load(open(f)); o = {k: dict(encode_s=v["encode_s"], decode_s=v["decode_s"]) for k, v in d.items() if "error" not in v}
    json.dump(o, open(f.replace("timingrun_", "timing_"), "w"), indent=1); print(f, len(o))
