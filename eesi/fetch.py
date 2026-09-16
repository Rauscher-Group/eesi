"""`eesi-fetch-data`: pull the third-party download and the checkpoints mirror.

Two independent pieces, run together by default or selected with `--osf` /
`--checkpoints`:

    --osf           the Klein, Kraemer & Noe 2023 OSF record (project `srqg7`):
                     `all_data_LJ13-1000.npy` -> `eesi.systems.lj13.data.REF_DATA_PATH`
                     `LJ13_eq_OT_flow_matching` -> `eesi.systems.lj13.dynamics.CKPT_PATH`
    --checkpoints   the companion checkpoints repo (see data/checkpoints/README.md)
                     into `data/checkpoints/`. Its URL does not exist yet -- set
                     `EESI_CHECKPOINTS_REPO` once it does.

Deliberately not run as part of `pip install` itself: network access during a
build/install step is fragile (editable installs, CI, sandboxed builds all tend to
break on it), so this is a separate command the README tells users to run afterward.
Uses only the standard library plus `tqdm` (already a dependency) -- no new dependency
just to download two files and clone a repo.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request

from tqdm import tqdm

from .paths import CHECKPOINTS_DIR

OSF_PROJECT_ID = "srqg7"
OSF_FILES_API = f"https://api.osf.io/v2/nodes/{OSF_PROJECT_ID}/files/osfstorage/"

# Placeholder until the checkpoints repo exists; fill this in (or set
# EESI_CHECKPOINTS_REPO) and `--checkpoints` starts working with no other change.
CHECKPOINTS_REPO_URL: str | None = None


def _osf_walk(api_url: str):
    """Yield (name, download_url) for every file under an OSF storage API URL, recursively."""
    while api_url:
        with urllib.request.urlopen(api_url) as resp:
            payload = json.load(resp)
        for entry in payload["data"]:
            attrs = entry["attributes"]
            if attrs["kind"] == "file":
                yield attrs["name"], entry["links"]["download"]
            elif attrs["kind"] == "folder":
                related = entry["relationships"]["files"]["links"]["related"]["href"]
                yield from _osf_walk(related)
        api_url = payload["links"].get("next")


def _osf_find(name: str) -> str:
    for fname, url in _osf_walk(OSF_FILES_API):
        if fname == name:
            return url
    raise SystemExit(
        f"{name!r} not found in OSF project {OSF_PROJECT_ID} "
        f"(https://osf.io/{OSF_PROJECT_ID}/) -- the record may have moved."
    )


def _download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "eesi-fetch-data"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)  # atomic: a crash mid-download never leaves a truncated dest


def fetch_osf() -> None:
    from eesi.systems.lj13.data import REF_DATA_PATH
    from eesi.systems.lj13.dynamics import CKPT_PATH

    for name, dest in {
        "all_data_LJ13-1000.npy": REF_DATA_PATH,
        "LJ13_eq_OT_flow_matching": CKPT_PATH,
    }.items():
        if dest.exists():
            print(f"{dest} already exists, skipping.")
            continue
        print(f"Locating {name} on OSF...")
        url = _osf_find(name)
        print(f"Downloading {name} -> {dest}")
        _download(url, dest)


def fetch_checkpoints_repo(dest: pathlib.Path = CHECKPOINTS_DIR) -> None:
    url = os.environ.get("EESI_CHECKPOINTS_REPO", CHECKPOINTS_REPO_URL)
    if not url:
        raise SystemExit(
            "No checkpoints repo is configured yet -- it hasn't been created. Once it "
            "exists, either set CHECKPOINTS_REPO_URL in eesi/fetch.py or run:\n"
            "    EESI_CHECKPOINTS_REPO=<git-url> eesi-fetch-data --checkpoints\n"
            "Until then, populate data/checkpoints/ by hand; see data/checkpoints/README.md."
        )
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", url, tmp], check=True)
        tmp_path = pathlib.Path(tmp)
        for item in tmp_path.iterdir():
            if item.name == ".git" or item.name.lower() == "readme.md":
                continue  # keep our own data/checkpoints/README.md as-is
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eesi-fetch-data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osf", action="store_true",
                        help="fetch the OSF LJ13 reference data + released checkpoint")
    parser.add_argument("--checkpoints", action="store_true",
                        help="clone the companion checkpoints repo into data/checkpoints/")
    args = parser.parse_args(argv)

    run_all = not (args.osf or args.checkpoints)
    if args.osf or run_all:
        fetch_osf()
    if args.checkpoints or run_all:
        fetch_checkpoints_repo()


if __name__ == "__main__":
    main()
