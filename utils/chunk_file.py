"""Cut a file into pieces then reassemble it later."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple
import zipfile

from tqdm import tqdm as _tqdm

DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB


def _pbar(total: int, desc: str):
    return _tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        smoothing=0.05,
    )


def parse_size(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("empty size string")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def sha256_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def read_in_chunks(f, chunk_size: int) -> Iterator[bytes]:
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        yield data


def _zip_compress_type() -> int:
    # Use deflate when available; fall back to stored
    try:
        return zipfile.ZIP_DEFLATED
    except Exception:
        return zipfile.ZIP_STORED


def split_file(
    src: Path,
    outdir: Path,
    chunk_size: int,
    prefix: str,
    manifest_name: str,
) -> Path:
    src = src.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = src.stem
    if manifest_name is None:
        manifest_name = f"{prefix}.manifest.json"

    total_size = src.stat().st_size
    total_parts = math.ceil(total_size / chunk_size) if total_size else 1
    parts: List[
        Tuple[str, str, int, int, str]
    ] = []  # (zip_filename, member_name, raw_size, comp_size, raw_sha256)
    comp_type = _zip_compress_type()

    if total_size == 0:
        # create a single empty zip chunk
        zip_name = f"{prefix}.part000001.zip"
        member_name = f"{prefix}.part000001.bin"
        zip_path = outdir / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=comp_type) as zf:
            info = zipfile.ZipInfo(
                member_name, date_time=datetime.now().timetuple()[:6]
            )
            info.compress_type = comp_type
            zf.writestr(info, b"")
        with zipfile.ZipFile(zip_path, "r") as zf:
            comp_size = zf.getinfo(member_name).compress_size
        parts.append(
            (
                zip_name,
                member_name,
                0,
                comp_size,
                hashlib.sha256(b"").hexdigest(),
            )
        )
    else:
        with src.open("rb") as f, _pbar(total_size, "Splitting (zip)") as pbar:
            for idx, data in enumerate(read_in_chunks(f, chunk_size), start=1):
                raw_sha = hashlib.sha256(data).hexdigest()
                zip_name = f"{prefix}.part{idx:06d}.zip"
                member_name = f"{prefix}.part{idx:06d}.bin"
                zip_path = outdir / zip_name
                with zipfile.ZipFile(zip_path, "w", compression=comp_type) as zf:
                    info = zipfile.ZipInfo(
                        member_name, date_time=datetime.now().timetuple()[:6]
                    )
                    info.compress_type = comp_type
                    zf.writestr(info, data)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    comp_size = zf.getinfo(member_name).compress_size
                parts.append((zip_name, member_name, len(data), comp_size, raw_sha))
                pbar.update(len(data))

    src_hash = sha256_file(src) if total_size else hashlib.sha256(b"").hexdigest()

    manifest = {
        "schema_version": 2,
        "mode": "zip_per_chunk",
        "original_filename": src.name,
        "original_size": total_size,
        "original_sha256": src_hash,
        "chunk_size": chunk_size,
        "total_parts": total_parts,
        "zip_compression": (
            "DEFLATED" if comp_type == zipfile.ZIP_DEFLATED else "STORED"
        ),
        "parts": [
            {
                "zip_filename": zf,
                "member_name": mem,
                "raw_size": rsz,
                "compressed_size": csz,
                "raw_sha256": rsha,
            }
            for (zf, mem, rsz, csz, rsha) in parts
        ],
    }

    manifest_path = outdir / manifest_name
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest_path


def join_file(
    manifest_path: Path,
    out_path: Path,
    verify_hashes: bool,
    bufsize: int = 1024 * 1024,
) -> Path:
    with manifest_path.open("r", encoding="utf-8") as mf:
        manifest = json.load(mf)

    manifest_dir = manifest_path.parent
    original_name = manifest["original_filename"]
    original_size = int(manifest["original_size"])
    original_sha256 = manifest["original_sha256"]
    parts = manifest["parts"]

    if out_path is None:
        out_path = manifest_dir / original_name
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as out_f, _pbar(original_size, "Joining (unzip)") as pbar:
        for entry in parts:
            zip_file = manifest_dir / entry["zip_filename"]
            member_name = entry["member_name"]
            if not zip_file.exists():
                raise FileNotFoundError(f"missing part archive: {zip_file}")
            with zipfile.ZipFile(zip_file, "r") as zf:
                if member_name not in zf.namelist():
                    raise FileNotFoundError(
                        f"member not found in {zip_file.name}: {member_name}"
                    )
                hasher = hashlib.sha256() if verify_hashes else None
                with zf.open(member_name, "r") as zmem:
                    while True:
                        chunk = zmem.read(bufsize)
                        if not chunk:
                            break
                        if hasher is not None:
                            hasher.update(chunk)
                        out_f.write(chunk)
                        pbar.update(len(chunk))
                if verify_hashes:
                    got = hasher.hexdigest()  # type: ignore[union-attr]
                    exp = entry["raw_sha256"]
                    if got != exp:
                        raise ValueError(
                            f"hash mismatch for {zip_file.name}:{member_name}: expected {exp}, got {got}"
                        )

    if verify_hashes:
        size = out_path.stat().st_size
        if size != original_size:
            raise ValueError(
                f"output size mismatch: expected {original_size}, got {size}"
            )
        out_hash = sha256_file(out_path, bufsize=bufsize)
        if out_hash != original_sha256:
            raise ValueError(
                f"output hash mismatch: expected {original_sha256}, got {out_hash}"
            )

    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="filechunker",
        description="Split a large file into zipped chunks and reassemble later.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="split a file into zipped chunk parts")
    sp.add_argument("src", type=Path, help="path to source file")
    sp.add_argument("outdir", type=Path, help="directory to write parts and manifest")
    sp.add_argument(
        "--chunk-size", type=str, default="64M", help="chunk size, e.g. 64M, 1G"
    )
    sp.add_argument(
        "--prefix", type=str, default=None, help="prefix for part filenames"
    )
    sp.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help="override manifest filename",
    )

    jp = sub.add_parser("join", help="join zipped parts using a manifest")
    jp.add_argument("manifest", type=Path, help="path to manifest json")
    jp.add_argument("--out", type=Path, default=None, help="output file path")
    jp.add_argument("--no-verify", action="store_true", help="skip sha256 verification")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "split":
        chunk_size = parse_size(args.chunk_size)
        manifest_path = split_file(
            src=args.src,
            outdir=args.outdir,
            chunk_size=chunk_size,
            prefix=args.prefix,
            manifest_name=args.manifest_name,
        )
        print(str(manifest_path))
    elif args.cmd == "join":
        out_path = join_file(
            manifest_path=args.manifest,
            out_path=args.out,
            verify_hashes=not args.no_verify,
        )
        print(str(out_path))


if __name__ == "__main__":
    main()
