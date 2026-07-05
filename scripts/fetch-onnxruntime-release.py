#!/usr/bin/env python3
"""
Download a prebuilt ONNX Runtime release package from GitHub for the current OS/CPU.

This avoids the ort-builder path when CPU EP is sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


GITHUB_API_BASE = "https://api.github.com/repos/microsoft/onnxruntime/releases"
USER_AGENT = "NeuralPlayer-DemucsONNX-Fetcher/1.0"


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent
    default_dest = script_root / "third_party" / "onnxruntime-release"

    parser = argparse.ArgumentParser(description="Fetch prebuilt ONNX Runtime package")
    parser.add_argument(
        "--version",
        default="latest-compatible",
        help="Release version (e.g. 1.20.1), 'latest', or 'latest-compatible' (default)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"Destination root (default: {default_dest})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite an existing extracted package",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for HTTPS requests (not recommended)",
    )
    return parser.parse_args()


def platform_asset_info() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in {"x86_64", "amd64"}:
            return "osx-x86_64", "tgz"
        if machine in {"arm64", "aarch64"}:
            return "osx-arm64", "tgz"
    elif system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "linux-x64", "tgz"
        if machine in {"arm64", "aarch64"}:
            return "linux-aarch64", "tgz"
    elif system == "windows":
        if machine in {"x86_64", "amd64"}:
            return "win-x64", "zip"
        if machine in {"arm64", "aarch64"}:
            return "win-arm64", "zip"

    raise RuntimeError(f"Unsupported platform/arch: system={system}, machine={machine}")


def release_url(version: str) -> str:
    if version == "latest":
        return f"{GITHUB_API_BASE}/latest"

    normalized = version if version.startswith("v") else f"v{version}"
    return f"{GITHUB_API_BASE}/tags/{normalized}"


def build_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def is_cert_verify_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
        if isinstance(reason, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(reason):
            return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def fetch_url_bytes(url: str, headers: dict[str, str], insecure: bool) -> bytes:
    req = urllib.request.Request(url, headers=headers)

    # 1) Standard system trust store.
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except Exception as exc:
        if not is_cert_verify_error(exc):
            raise

    # 2) Python certifi CA bundle, if available.
    try:
        import certifi  # type: ignore

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, context=context) as response:
            return response.read()
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            pass
        elif not is_cert_verify_error(exc):
            # Non-SSL failure, bubble it up.
            raise

    # 3) Explicitly insecure mode.
    if insecure:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response:
            return response.read()

    # 4) Fallback to curl, which may have access to a working OS trust store.
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-fsSL", url]
        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])
        result = subprocess.run(cmd, capture_output=True, text=False)
        if result.returncode == 0:
            return result.stdout

    raise RuntimeError(
        "TLS certificate verification failed. "
        "Try installing certifi (`python -m pip install certifi`) or rerun with --insecure."
    )


def request_json(url: str, insecure: bool) -> dict:
    payload = fetch_url_bytes(url, build_headers(), insecure)
    return json.loads(payload.decode("utf-8"))


def request_json_list(url: str, insecure: bool) -> list[dict]:
    payload = fetch_url_bytes(url, build_headers(), insecure)
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list from {url}")
    return data


def find_asset(release_data: dict, platform_key: str, ext: str) -> dict:
    tag = release_data.get("tag_name", "")
    version = tag[1:] if tag.startswith("v") else tag
    expected_name = f"onnxruntime-{platform_key}-{version}.{ext}"

    assets = release_data.get("assets", [])

    for asset in assets:
        if asset.get("name") == expected_name:
            return asset

    for asset in assets:
        name = asset.get("name", "")
        if name.startswith(f"onnxruntime-{platform_key}-") and name.endswith(f".{ext}"):
            return asset

    available = [asset.get("name", "<unnamed>") for asset in assets if "onnxruntime-" in asset.get("name", "")]
    raise RuntimeError(
        "Could not find a matching ONNX Runtime asset.\n"
        f"Expected: {expected_name}\n"
        "Available ONNX Runtime assets:\n"
        + "\n".join(f"  - {name}" for name in available)
    )


def resolve_release_and_asset(version: str, platform_key: str, ext: str, insecure: bool) -> tuple[dict, dict]:
    if version == "latest-compatible":
        page = 1
        per_page = 100
        while True:
            release_list_url = f"{GITHUB_API_BASE}?per_page={per_page}&page={page}"
            releases = request_json_list(release_list_url, insecure)
            if not releases:
                break

            for release_data in releases:
                if release_data.get("draft") or release_data.get("prerelease"):
                    continue
                try:
                    return release_data, find_asset(release_data, platform_key, ext)
                except RuntimeError:
                    continue

            page += 1

        raise RuntimeError(
            f"Could not find any release asset for platform '{platform_key}' with extension '.{ext}'. "
            "Try specifying an explicit version with --version."
        )

    release_data = request_json(release_url(version), insecure)
    return release_data, find_asset(release_data, platform_key, ext)


def download_file(url: str, destination: Path, insecure: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_url_bytes(url, {"User-Agent": USER_AGENT, **build_headers()}, insecure)
    destination.write_bytes(payload)


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)

    entries = [p for p in output_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return output_dir


def update_current_link(dest_root: Path, extracted_root: Path) -> None:
    current = dest_root / "current"
    if current.is_symlink() or current.exists():
        if current.is_dir() and not current.is_symlink():
            shutil.rmtree(current)
        else:
            current.unlink()

    try:
        current.symlink_to(extracted_root, target_is_directory=True)
    except OSError:
        shutil.copytree(extracted_root, current)


def main() -> int:
    args = parse_args()

    try:
        platform_key, ext = platform_asset_info()
        release_data, asset = resolve_release_and_asset(args.version, platform_key, ext, args.insecure)
    except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Failed to resolve release asset: {exc}", file=sys.stderr)
        return 1

    tag = release_data.get("tag_name", "")
    version = tag[1:] if tag.startswith("v") else tag
    if not version:
        print("Failed to determine ONNX Runtime version from release metadata.", file=sys.stderr)
        return 1

    asset_name = asset["name"]
    download_url = asset["browser_download_url"]
    stem = asset_name[: -len(f".{ext}")]

    dest_root = args.dest.resolve()
    versions_root = dest_root / "versions"
    extracted_dir = versions_root / stem

    if extracted_dir.exists() and not args.force:
        extracted_root = extracted_dir
        if len(list(extracted_dir.iterdir())) == 1:
            only_entry = next(extracted_dir.iterdir())
            if only_entry.is_dir():
                extracted_root = only_entry
        update_current_link(dest_root, extracted_root)
        print(f"Reusing existing ONNX Runtime package: {extracted_root}")
    else:
        with tempfile.TemporaryDirectory(prefix="onnxruntime-release-") as temp_dir:
            archive_path = Path(temp_dir) / asset_name
            print(f"Downloading {asset_name}...")
            try:
                download_file(download_url, archive_path, args.insecure)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"Failed to download asset: {exc}", file=sys.stderr)
                return 1
            except RuntimeError as exc:
                print(f"Failed to download asset: {exc}", file=sys.stderr)
                return 1

            print(f"Extracting to {extracted_dir}...")
            extracted_root = extract_archive(archive_path, extracted_dir)
            update_current_link(dest_root, extracted_root)

    current_root = (dest_root / "current").resolve()
    include_dir = current_root / "include"
    lib_dir = current_root / "lib"

    if not include_dir.exists() or not lib_dir.exists():
        print(
            "Downloaded package is missing include/ or lib/ directories.\n"
            f"Package root: {current_root}",
            file=sys.stderr,
        )
        return 1

    print("")
    print("ONNX Runtime package ready:")
    print(f"  Version: {version}")
    print(f"  Root:    {current_root}")
    print("")
    print("Use this CMake argument for Demucs CLI:")
    print(f"  -DDEMUCS_ONNXRUNTIME_ROOT={current_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
