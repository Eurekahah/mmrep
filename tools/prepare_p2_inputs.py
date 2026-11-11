"""Download and verify the Flickr/COCO inputs used by P2 (CPU only).

Use ``fetch-coco`` and ``fetch-flickr`` from the repository root with
``python -m tools.prepare_p2_inputs``. They may run concurrently. Then
run ``verify`` to unpack, regenerate both P1 splits, decode every val/test
image and publish logs/p2_data_validation.json. ``watch-verify`` performs
the same checks as each download becomes available. The transfer directory
keeps interrupted ranges for retry until the caller removes verified copies.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
import zipfile
import zlib

from data.config import get_paths


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(get_paths()[0])
LOGS = ROOT / "logs"
FLICKR = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/"
COCO = "http://images.cocodataset.org/"
TFDS = "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt"
FILES = {
    "flickr30k-images.zip": ("flickr30k", FLICKR + "flickr30k-images.zip", 4390240817,
        "2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391"),
    "flickr_annotations_30k.csv": ("flickr30k", FLICKR + "flickr_annotations_30k.csv", 12896305,
        "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137"),
    "val2017.zip": ("coco", COCO + "zips/val2017.zip", 815585330,
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"),
    "annotations_trainval2017.zip": ("coco", COCO + "annotations/annotations_trainval2017.zip", 252907541,
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"),
}


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ranged_part(name, start, end, chunk_root):
    _, url, size, _ = FILES[name]
    path = chunk_root / f"{start:012d}-{end:012d}.part"
    earlier = chunk_root.parent.parent / FILES[name][0] / (name + ".part")
    if not path.exists() and earlier.is_file() and earlier.stat().st_size > end:
        with earlier.open("rb") as source:
            source.seek(start)
            block = source.read(end - start + 1)
        if len(block) == end - start + 1:
            path.write_bytes(block)
            return path
    for attempt in range(6):
        already = path.stat().st_size if path.exists() else 0
        length = end - start + 1
        if already == length:
            return path
        if already > length:
            raise ValueError(f"Oversize segment: {path}")
        first = start + already
        req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{end}", "Accept-Encoding": "identity"})
        try:
            if FILES[name][0] == "flickr30k":
                fragment = path.with_suffix(".response")
                try:
                    # Use the native TLS stack on Windows. Never log the signed redirect.
                    call = subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-fsSL",
                        "--connect-timeout", "15", "--max-time", "180", "--range", f"{first}-{end}",
                        "--output", str(fragment), "--write-out", "%{http_code} %header{content-range}", url],
                        capture_output=True, text=True, errors="replace")
                    if call.stdout.strip() != f"206 bytes {first}-{end}/{size}":
                        raise OSError(f"No matching HTTP Range response for {name}")
                    received = fragment.stat().st_size if fragment.exists() else 0
                    if received > end - first + 1:
                        raise ValueError(f"Unexpected extra bytes in Range response: {name}")
                    if received:
                        with fragment.open("rb") as response, path.open("ab") as dest:
                            shutil.copyfileobj(response, dest, length=1024 * 1024)
                    if path.exists() and path.stat().st_size == length:
                        return path
                    raise EOFError("Range response incomplete")
                finally:
                    fragment.unlink(missing_ok=True)
            with urllib.request.urlopen(req, timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{end}/{size}":
                    raise ValueError(f"Source did not confirm the requested byte interval for {name}")
                with path.open("ab") as dest:
                    remaining = length - already
                    while remaining:
                        block = response.read(min(65536, remaining))
                        if not block:
                            raise EOFError("Range response ended early")
                        dest.write(block)
                        remaining -= len(block)
                if path.stat().st_size == length:
                    return path
        except (OSError, EOFError, http.client.HTTPException) as exc:
            if attempt == 5:
                raise RuntimeError(f"{name} segment {start}-{end} failed ({type(exc).__name__})") from None
            time.sleep(min(2 ** attempt, 10))
    raise OSError(f"Incomplete segment: {path}")


def fetch_ranges(data, workers, names):
    """Resume verified Range downloads, then hash and assemble each archive."""
    dataset_dir = data / FILES[names[0]][0]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    size_per_part = (8 if FILES[names[0]][0] == "flickr30k" else 2) * 1024 * 1024
    items = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inputs = {}
        for name in names:
            _, _, size, expected = FILES[name]
            target = dataset_dir / name
            if target.is_file() and target.stat().st_size == size and sha256(target) == expected:
                continue
            directory = data / ".transfers" / name
            directory.mkdir(parents=True, exist_ok=True)
            inputs[name] = [(name, start, min(start + size_per_part, size) - 1, directory)
                            for start in range(0, size, size_per_part)]
        # Interleave the smaller annotation transfer and image transfer so both advance.
        for i in range(max(map(len, inputs.values()), default=0)):
            for name, pieces in inputs.items():
                if i < len(pieces):
                    f = pool.submit(_ranged_part, *pieces[i])
                    items[f] = (name, i)
        finished = {name: {} for name in inputs}
        for future in as_completed(items):
            name, i = items[future]
            finished[name][i] = future.result()
            count = len(finished[name])
            total = len(inputs[name])
            if count == total or count % 20 == 0:
                print(f"{name}: {count}/{total} ranges complete", flush=True)
            if count != total:
                continue
            joined = dataset_dir / (name + ".assembling")
            with joined.open("wb") as dest:
                for index in range(total):
                    with finished[name][index].open("rb") as source:
                        shutil.copyfileobj(source, dest, length=1024 * 1024)
            expected_size, expected_sha = FILES[name][2:]
            if joined.stat().st_size != expected_size or sha256(joined) != expected_sha:
                raise ValueError(f"Assembled archive failed size/SHA256 verification: {name}")
            os.replace(joined, dataset_dir / name)
            print(f"{name}: archived and SHA256 verified", flush=True)


def fetch_flickr(data, workers):
    def fetch(name):
        subdir, url, size, expected_sha = FILES[name]
        dest = data / subdir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size == size and sha256(dest) == expected_sha:
                return
            raise ValueError(f"Existing file fails verification: {dest}")
        part = dest.with_name(name + ".part")
        for attempt in range(12):
            length = part.stat().st_size if part.exists() else 0
            if length == size:
                if sha256(part) != expected_sha:
                    raise ValueError(f"SHA256 mismatch: {part}")
                os.replace(part, dest)
                return
            if length > size:
                raise ValueError(f"Oversize partial file: {part}")
            print(f"{name}: resuming at {length} bytes", flush=True)
            subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-fsSL", "--connect-timeout", "15",
                            "--max-time", "1800", "--continue-at", "-", "--output", str(part), url], check=False)
            if attempt < 11 and part.exists() and part.stat().st_size != size:
                time.sleep(3)
        if part.exists() and part.stat().st_size == size and sha256(part) == expected_sha:
            os.replace(part, dest)
            return
        raise OSError(f"Incomplete archive after retries: {name}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        csv = pool.submit(fetch, "flickr_annotations_30k.csv")
        image_zip = pool.submit(fetch_ranges, data, workers, ("flickr30k-images.zip",))
        csv.result()
        image_zip.result()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        tmp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def _files_for(dataset):
    folder = "flickr30k" if dataset == "flickr" else "coco"
    return [name for name, spec in FILES.items() if spec[0] == folder]


def _arrived(name, data):
    subdir, _, size, _ = FILES[name]
    path = data / subdir / name
    # Download workers publish this name only after the bytes and hash match.
    # A .part with the right size can still be open in another verifier.
    return path.is_file() and path.stat().st_size == size


def _verified_source(name, data):
    subdir, url, size, expected = FILES[name]
    official = data / subdir / name
    path = official if official.exists() else official.with_name(name + ".part")
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing or incomplete {name}: expected {size} bytes")
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"SHA256 mismatch for {name}: {digest} != {expected}")
    if path != official:
        os.replace(path, official)
    reference = ("https://huggingface.co/datasets/nlphuji/flickr30k/blob/main/" + name
                 if subdir == "flickr30k" else TFDS)
    return official, {"filename": name, "source": url, "bytes": size, "sha256": digest,
                      "checksum_reference": reference, "sha256_matches_published": True}


def _crc(path):
    val = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            val = zlib.crc32(chunk, val)
    return val & 0xffffffff


def _extract(archive, name, data, first_ids):
    subdir = FILES[name][0]
    root = (data / subdir).resolve()
    is_annotations = name == "annotations_trainval2017.zip"
    folder = {"flickr30k-images.zip": "flickr30k-images", "val2017.zip": "val2017",
              "annotations_trainval2017.zip": "annotations"}[name]
    out_dir = (root / folder).resolve()
    if not out_dir.is_relative_to(root):
        raise ValueError(f"Extraction target outside data root: {out_dir}")
    out_dir.mkdir(exist_ok=True)
    annotations = {"instances_val2017.json", "captions_val2017.json"}
    with zipfile.ZipFile(archive) as z:
        bad_member = z.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failed: {name}, member {bad_member}")
        selected = []
        for member in z.infolist():
            member_path = PurePosixPath(member.filename)
            if (len(member_path.parts) != 2 or member_path.parts[0] != folder or member.is_dir()):
                continue
            leaf = member_path.name
            if (leaf in annotations if is_annotations else leaf.lower().endswith(".jpg")):
                if "\\" in leaf or ":" in leaf or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Unexpected archive member: {member.filename}")
                selected.append((leaf, member))
        if not selected or (is_annotations and {leaf for leaf, _ in selected} != annotations):
            raise ValueError(f"Missing expected archive members for {name}")
        selected.sort(key=lambda item: item[0] not in first_ids)
        for leaf, member in selected:
            target = out_dir / leaf
            if target.exists():
                if target.is_symlink() or target.stat().st_size != member.file_size or _crc(target) != member.CRC:
                    raise ValueError(f"Existing extracted file differs from verified archive: {target}")
                continue
            with tempfile.NamedTemporaryFile(dir=out_dir, delete=False, suffix=".extracting") as out:
                tmp = Path(out.name)
                try:
                    with z.open(member) as source:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                except BaseException:
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise
            if tmp.stat().st_size != member.file_size:
                tmp.unlink()
                raise ValueError(f"Short extracted file: {member.filename}")
            os.replace(tmp, target)
        return {"zip_crc_all_passed": True, "zip_members_checked": len(z.infolist()),
                "members_materialized": len(selected), "extracted_directory": str(out_dir)}


def verify_dataset(dataset, data, logs):
    from PIL import Image
    from data.prepare_data import prepare_coco, prepare_flickr
    from tools.p2_image_identity import image_bytes_sha256

    src_split = ROOT / "splits" / (dataset + "_split.json")
    split_bytes = src_split.read_bytes()
    meta = json.loads(split_bytes)
    expected_per_split = {"flickr": 1000, "coco": 2476}[dataset]
    if any(len(meta[key]) != expected_per_split for key in ("val", "test")):
        raise ValueError(f"Unexpected official P2 val/test size for {dataset}")
    first_ids = set(meta["val"]) | set(meta["test"])
    archives = []
    for name in _files_for(dataset):
        archive, record = _verified_source(name, data)
        if archive.suffix == ".zip":
            print(f"{dataset}: verifying all ZIP CRCs and unpacking {name}", flush=True)
            record.update(_extract(archive, name, data, first_ids))
        archives.append(record)
    receipts = logs / "p2_data_sources"
    receipts.mkdir(parents=True, exist_ok=True)
    regenerated = receipts / (dataset + "_split.regenerated.json")
    if dataset == "flickr":
        img_dir = data / "flickr30k" / "flickr30k-images"
        prepare_flickr(str(data / "flickr30k" / "flickr_annotations_30k.csv"), str(regenerated), images=str(img_dir))
    else:
        img_dir = data / "coco" / "val2017"
        prepare_coco(str(data / "coco"), str(regenerated), images=str(img_dir))
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    regen_bytes = regenerated.read_bytes()
    if regen_bytes != split_bytes:
        raise ValueError(f"P1 regenerated {dataset} split differs byte-for-byte from checked-in split")

    def decode(img_id):
        if len(meta["captions"][img_id]) != 5 or img_id not in meta["labels"]:
            raise ValueError(f"Missing captions or labels for {img_id}")
        with Image.open(img_dir / img_id) as image:
            image.load()
            image.convert("RGB").load()
        return 1

    splits = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key in ("val", "test"):
            ids = meta[key]
            decoded = sum(pool.map(decode, ids))
            splits[key] = {"images": len(ids), "decoded": decoded, "captions_per_image": 5,
                           "image_bytes_sha256": image_bytes_sha256(img_dir, ids)}
    if len(first_ids) != sum(item["images"] for item in splits.values()):
        raise ValueError(f"Non-unique IDs across {dataset} val/test")
    result = {"dataset": dataset, "status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(data.resolve()), "archives": archives, "splits": splits,
              "decoded_total": sum(s["decoded"] for s in splits.values()),
              "checked_in_split": str(src_split), "checked_in_split_sha256": split_hash,
              "regenerated_split": str(regenerated), "regenerated_split_sha256": hashlib.sha256(regen_bytes).hexdigest(),
              "split_bytes_identical": True}
    save_json(receipts / (dataset + ".json"), result)
    print(f"{dataset}: PASSED, {result['decoded_total']} real val/test images decoded, split bytes identical", flush=True)
    return result


def verify_all(data, logs, wait):
    datasets = ("coco", "flickr")
    waiting = set(datasets)
    results = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while waiting or futures:
            for d in list(waiting):
                if wait and not all(_arrived(n, data) for n in _files_for(d)):
                    continue
                futures[pool.submit(verify_dataset, d, data, logs)] = d
                waiting.remove(d)
            for f in list(futures):
                if f.done():
                    results[futures.pop(f)] = f.result()
            if waiting or futures:
                time.sleep(2)
    result = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(data.resolve()), "datasets": {d: results[d] for d in datasets},
              "decoded_total": sum(r["decoded_total"] for r in results.values())}
    save_json(logs / "p2_data_validation.json", result)
    print(f"P2 DATA PASSED {result['decoded_total']} source images. Receipt: {logs / 'p2_data_validation.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch-coco", "fetch-flickr", "verify", "watch-verify"))
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    parser.add_argument("--workers", type=int, default=24, help="Concurrent archive range transfers")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.action == "fetch-coco":
        fetch_ranges(args.data_root, args.workers, ("val2017.zip", "annotations_trainval2017.zip"))
    elif args.action == "fetch-flickr":
        fetch_flickr(args.data_root, args.workers)
    else:
        verify_all(args.data_root, args.log_dir, args.action == "watch-verify")


if __name__ == "__main__":
    main()
