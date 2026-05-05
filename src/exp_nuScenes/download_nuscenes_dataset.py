"""Download and extract nuScenes dataset archives from a config file.

Expected config section:

dataset_download:
  save_root: dataset/nuScenes
  archives:
    - url: "https://...signed..."
      filename: "v1.0-trainval01_blobs.tgz"  # optional
      extract: true                            # optional, default true
      sha256: "..."                           # optional
      extract_to: "dataset/nuScenes"          # optional
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exp_driving_videos.pipe_utils.exp_driving_utils import load_pattern_cfg_file


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_nuScenes" / "default.yaml"
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _infer_filename(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        raise ValueError(f"Unable to infer filename from URL: {url}")
    return name


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".zip")
        or name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar.bz2")
        or name.endswith(".tbz2")
        or name.endswith(".tar.xz")
        or name.endswith(".txz")
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_dir)):
                raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
        tf.extractall(target_dir)


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            member_path = (target_dir / name).resolve()
            if not str(member_path).startswith(str(target_dir)):
                raise RuntimeError(f"Unsafe zip member path detected: {name}")
        zf.extractall(target_dir)


def _download_with_resume(url: str, output_path: Path, force: bool = False, timeout_sec: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"[skip] already downloaded: {output_path}")
        return output_path

    part_path = Path(str(output_path) + ".part")
    if force and part_path.exists():
        part_path.unlink()

    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "CauVid-nuScenes-downloader/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = Request(url, headers=headers)
    print(f"[download] {url}")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            status_code = getattr(resp, "status", 200)
            supports_resume = status_code == 206
            if existing > 0 and not supports_resume:
                existing = 0
                mode = "wb"
            else:
                mode = "ab" if existing > 0 else "wb"

            with part_path.open(mode) as out:
                while True:
                    chunk = resp.read(DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    except PermissionError as exc:
        # Common after a previous root-run download left an unwritable *.part file.
        if part_path.exists() and not os.access(part_path, os.W_OK):
            raise PermissionError(
                f"Cannot write to '{part_path}'. The file is not writable by the current user. "
                "Fix ownership/permissions of the nuScenes download directory and retry."
            ) from exc
        raise

    shutil.move(str(part_path), str(output_path))
    print(f"[ok] saved: {output_path}")
    return output_path


def _extract_archive(archive_path: Path, extract_to: Path, force: bool = False) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    stamp = extract_to / f".extract_done_{archive_path.name}.stamp"
    if stamp.exists() and not force:
        print(f"[skip] already extracted: {archive_path.name}")
        return

    if archive_path.suffix.lower() == ".zip":
        print(f"[extract] zip -> {extract_to}")
        _safe_extract_zip(archive_path, extract_to)
    elif _is_archive(archive_path):
        print(f"[extract] tar -> {extract_to}")
        _safe_extract_tar(archive_path, extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    stamp.write_text("ok\n", encoding="utf-8")
    print(f"[ok] extracted: {archive_path.name}")


def _iter_archives(download_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    archives = download_cfg.get("archives", [])
    if not isinstance(archives, list):
        raise ValueError("dataset_download.archives must be a list")
    for index, item in enumerate(archives):
        if not isinstance(item, dict):
            raise ValueError(f"dataset_download.archives[{index}] must be a mapping")
        if not item.get("url"):
            raise ValueError(f"dataset_download.archives[{index}] missing required key 'url'")
        yield item


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract nuScenes archives from config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path")
    parser.add_argument("--list-only", action="store_true", help="Only print planned downloads")
    parser.add_argument("--no-extract", action="store_true", help="Download only, skip extraction")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract even if present")
    parser.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout seconds")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_pattern_cfg_file(args.config) or {}
    download_cfg = cfg.get("dataset_download", {})
    if not isinstance(download_cfg, dict):
        raise ValueError("'dataset_download' must be a mapping in config")

    save_root = _resolve_project_path(str(download_cfg.get("save_root", "dataset/nuScenes")))
    save_root.mkdir(parents=True, exist_ok=True)

    archives = list(_iter_archives(download_cfg))
    if not archives:
        print("No archives configured. Set dataset_download.archives in config and re-run.")
        return

    print(f"Config: {args.config}")
    print(f"Save root: {save_root}")
    print(f"Archives: {len(archives)}")

    for i, item in enumerate(archives, start=1):
        url = str(item["url"])
        filename = str(item.get("filename") or _infer_filename(url))
        output_path = save_root / filename
        extract_flag = bool(item.get("extract", True)) and (not args.no_extract)
        extract_to_cfg = item.get("extract_to")
        extract_to = _resolve_project_path(str(extract_to_cfg)) if extract_to_cfg else save_root
        sha256_expected = item.get("sha256")

        print(f"[{i}/{len(archives)}] {filename}")
        if args.list_only:
            print(f"  url={url}")
            print(f"  save={output_path}")
            print(f"  extract={extract_flag} -> {extract_to}")
            continue

        downloaded = _download_with_resume(
            url=url,
            output_path=output_path,
            force=args.force,
            timeout_sec=int(args.timeout_sec),
        )

        if sha256_expected:
            digest = _sha256(downloaded)
            if digest.lower() != str(sha256_expected).lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {downloaded.name}: expected {sha256_expected}, got {digest}"
                )
            print(f"[ok] sha256 verified: {downloaded.name}")

        if extract_flag:
            _extract_archive(downloaded, extract_to, force=args.force)

    print("nuScenes dataset download workflow completed.")


if __name__ == "__main__":
    main()
