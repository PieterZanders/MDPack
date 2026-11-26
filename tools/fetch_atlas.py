"""Fetch only the reference PDB and replica-1 trajectory of an ATLAS entry using HTTP range requests
on the public protein zip (https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/<id>/<id>_protein.zip)."""
import os, struct, sys, zlib, urllib.request

def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=600).read()

def members(url):
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
    tail = rng(url, max(0, size - 65536), size - 1)
    e = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    cd = rng(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode()
        out[name] = (off, csize, meth)
        p += 46 + nl + el + cl
    return out

def get(url, info, dst):
    off, csize, meth = info
    hdr = rng(url, off, off + 29)
    nl, el = struct.unpack("<HH", hdr[26:30])
    start = off + 30 + nl + el
    data = b"".join(rng(url, s, min(s + 8_000_000, start + csize) - 1) for s in range(start, start + csize, 8_000_000))
    data = zlib.decompress(data, -15) if meth == 8 else data
    open(dst, "wb").write(data)

if __name__ == "__main__":
    pid, outdir = sys.argv[1], sys.argv[2]
    url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pid}/{pid}_protein.zip"
    m = members(url)
    os.makedirs(outdir, exist_ok=True)
    for name in [f"{pid}.pdb", f"{pid}_prod_R1_fit.xtc", "README.txt"]:
        if name in m and not os.path.exists(os.path.join(outdir, name)):
            get(url, m[name], os.path.join(outdir, name))
            print(pid, "got", name, flush=True)
