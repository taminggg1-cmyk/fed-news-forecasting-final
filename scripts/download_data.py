"""Download the final study's training data from this repository's GitHub release and unpack it into data/.

Run from the repository root:  uv run python scripts/download_data.py
(plain `python` works too: it uses the standard library only)

Downloads final_study_data.zip (79 MB), unpacks its 53 files into data/, and checks each against the sha256 in
MANIFEST.txt. config.py then finds data/ automatically. Run it again to re-check; files already correct are kept.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://github.com/taminggg1-cmyk/fed-news-forecasting-final/releases/download/data-2026-10-18/"
       "final_study_data.zip")
TOP = "fed_news_data/"  # folder name inside the zip


def sha256(path: Path) -> str:
    """sha256 of a file, read in 1 MB pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(dest: Path) -> dict[str, str]:
    """{relative path: sha256} from dest/MANIFEST.txt."""
    rows = (dest / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()[1:]
    return {line.split()[2]: line.split()[0] for line in rows if line.strip()}


def check(dest: Path) -> list[str]:
    """Files listed in the manifest that are missing or whose checksum differs."""
    if not (dest / "MANIFEST.txt").exists():
        return ["MANIFEST.txt"]
    return [rel for rel, digest in manifest(dest).items()
            if not (dest / rel).exists() or sha256(dest / rel) != digest]


def download(url: str, target: Path) -> None:
    """Download url to target, printing progress."""
    with urllib.request.urlopen(url) as resp, open(target, "wb") as out:
        total, done = int(resp.headers.get("Content-Length", 0)), 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                print(f"\rdownloading {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def unpack(zip_path: Path, dest: Path) -> None:
    """Extract every file under fed_news_data/ in the zip into dest (never outside it)."""
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(TOP):
                continue
            out = (dest / info.filename[len(TOP):]).resolve()
            if root not in out.parents:
                raise ValueError(f"unsafe path in zip: {info.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main(argv=None) -> int:
    """Download and unpack unless data/ already matches the manifest; return 0 when every file checks out."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "data"), help="target folder")
    ap.add_argument("--zip", help="use an already downloaded final_study_data.zip instead of downloading")
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    if not check(dest):
        print(f"{dest}: all files present and correct")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip) if args.zip else dest / "final_study_data.zip"
    if not args.zip:
        download(URL, zip_path)
    unpack(zip_path, dest)
    if not args.zip:
        zip_path.unlink()
    bad = check(dest)
    if bad:
        print(f"checksum mismatch or missing: {bad}")
        return 1
    print(f"{dest}: {len(manifest(dest))} files unpacked and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
