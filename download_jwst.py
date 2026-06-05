"""
JWST / MAST image downloader for sft_torch pipeline.
Downloads level-3 calibrated NIRCam images (i2d.fits) from MAST.
"""
import requests, sys, os, time, argparse
from urllib.parse import quote

MAST_BASE = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"

KNOWN_PROGRAMS = {
    "smacs":  {
        "pid": 2736, "name": "SMACS 0723 (cluster + lensing)",
        "obs": ["001"], "filters": ["f090w", "f150w", "f200w"],
    },
    "ceers":  {
        "pid": 2756, "name": "CEERS (wide galaxy survey)",
        "obs": ["002", "003"], "filters": ["f115w", "f150w", "f200w", "f277w", "f356w", "f444w"],
    },
    "hudf":   {
        "pid": None, "name": "HUDF (ultra-deep field) — external file",
        "obs": [], "filters": [],
    },
    "nebula": {
        "pid": 2733, "name": "Southern Ring Nebula",
        "obs": ["001"], "filters": ["f090w", "f187n", "f356w", "f444w"],
    },
}


def probe_file(pid, obs, filt):
    uri = f"jw{pid:05d}-o{obs}_t001_nircam_clear-{filt}_i2d.fits"
    url = MAST_BASE + uri
    try:
        r = requests.head(url, timeout=10)
        if r.status_code == 200:
            sz = int(r.headers.get("Content-Length", 0)) / 1e6
            return uri, sz
    except Exception:
        pass
    return None, 0


def download_file(uri, outdir=".", timeout=600):
    url = MAST_BASE + uri
    fname = uri if "/" not in uri else uri.split("/")[-1]
    outpath = os.path.join(outdir, fname)
    if os.path.exists(outpath) and os.path.getsize(outpath) > 10000:
        print(f"  SKIP (exists): {fname}")
        return outpath
    print(f"  DOWNLOAD {fname} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    r = requests.get(url, stream=True, timeout=timeout)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}")
        return None
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(262144):
            f.write(chunk)
    sz = os.path.getsize(outpath) / 1e6
    dt = time.perf_counter() - t0
    print(f"{sz:.0f}MB {dt:.0f}s")
    return outpath


def list_available(program):
    info = KNOWN_PROGRAMS.get(program)
    if not info:
        print(f"Unknown program: {program}")
        print(f"Known: {list(KNOWN_PROGRAMS.keys())}")
        return
    if info["pid"] is None:
        print(f"{info['name']}: external file, no MAST download")
        return
    print(f"{info['name']} (PID {info['pid']}):")
    for obs in info["obs"]:
        for filt in info["filters"]:
            uri, sz = probe_file(info["pid"], obs, filt)
            if uri:
                print(f"  {uri}  ({sz:.0f}MB)")
            else:
                print(f"  jw{info['pid']:05d}-o{obs}_t001_nircam_clear-{filt}_i2d.fits  NOT FOUND")


def download_program(program, outdir=".", filt=None, obs=None):
    info = KNOWN_PROGRAMS.get(program)
    if not info or info["pid"] is None:
        print(f"Cannot download: {program}")
        return []
    obs_list = [obs] if obs else info["obs"]
    filt_list = [filt] if filt else info["filters"]
    os.makedirs(outdir, exist_ok=True)
    downloaded = []
    for o in obs_list:
        for f in filt_list:
            uri, sz = probe_file(info["pid"], o, f)
            if uri:
                path = download_file(uri, outdir)
                if path:
                    downloaded.append(path)
            else:
                print(f"  NOT FOUND: jw{info['pid']:05d}-o{o}_t001_nircam_clear-{f}_i2d.fits")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download JWST images from MAST")
    parser.add_argument("program", nargs="?", default="list",
                        help="Program name or 'list'")
    parser.add_argument("--filter", dest="filt", help="Specific filter (e.g. f090w)")
    parser.add_argument("--obs", help="Specific observation (e.g. 001)")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available files")
    args = parser.parse_args()

    if args.program == "list":
        for p in KNOWN_PROGRAMS:
            print(f"\n{p}: {KNOWN_PROGRAMS[p]['name']}")
        return

    if args.list:
        list_available(args.program)
        return

    download_program(args.program, args.outdir, args.filt, args.obs)


if __name__ == "__main__":
    main()
