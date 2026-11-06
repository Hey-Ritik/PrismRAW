"""Download one capture per physical scene from a pinned SIDD-CC revision.

Small pilot only, not an official benchmark. Refer to the dataset card for terms.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "MhdMohammadi/SIDD-CC"
REVISION = "5eb2ab6375d7f99f36b59bb5aacb46af18409412"


def request_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=3, choices=range(3, 11))
    parser.add_argument("--diverse-cameras", action="store_true", help="Choose varied cameras for expanded training scenes")
    args = parser.parse_args()
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/"
    directories = request_json(api + "Raw/train/Data?limit=1000")
    groups = {}
    for entry in sorted(directories, key=lambda item: item["path"]):
        if entry["type"] == "directory":
            name = entry["path"].split("/")[-1]
            groups.setdefault(name.split("_")[1], entry["path"])
    if args.diverse_cameras:
        preferences = {"004": "GP", "005": "IP", "006": "G4", "007": "S6"}
        for scene, camera in preferences.items():
            candidates = [entry["path"] for entry in directories if entry["type"] == "directory"
                          and entry["path"].split("/")[-1].split("_")[1:3] == [scene, camera]]
            if candidates:
                groups[scene] = sorted(candidates)[0]
    root = Path("data/raw/sidd-cc-subset")
    records = []
    for scene_id, directory in sorted(groups.items())[:args.scenes]:
        entries = request_json(api + directory)
        by_name = {entry["path"].split("/")[-1]: entry for entry in entries}
        noisy_name = sorted(name for name in by_name if "_NOISY_RAW_" in name)[0]
        names = [noisy_name, noisy_name.replace("_NOISY_", "_GT_"),
                 noisy_name.replace("_NOISY_", "_METADATA_")]
        record = {"scene_id": scene_id, "instance": directory.split("/")[-1], "files": {}}
        for kind, name in zip(("noisy", "clean", "metadata"), names):
            info = by_name[name]
            target = root / record["instance"] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_hash = info.get("lfs", {}).get("oid")
            valid = target.exists() and target.stat().st_size == info["size"]
            if valid and expected_hash:
                valid = sha256(target) == expected_hash
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{info['path']}"
            if not valid:
                print(f"Downloading {name} ({info['size'] / 1e6:.1f} MB)", flush=True)
                temporary = target.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                if temporary.stat().st_size != info["size"]:
                    raise ValueError(f"Incomplete download: {name}")
                if expected_hash and sha256(temporary) != expected_hash:
                    raise ValueError(f"Checksum mismatch: {name}")
                temporary.replace(target)
            record["files"][kind] = {"path": target.as_posix(), "url": url,
                                      "sha256": sha256(target), "bytes": target.stat().st_size}
            print(f"Verified {name}", flush=True)
        records.append(record)
        index = {"dataset": REPO, "revision": REVISION,
                 "terms": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
                 "purpose": "Small scene-separated pilot; not an official benchmark", "records": records}
        (root / "pilot-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    if len(records) != args.scenes:
        raise ValueError("Not enough physical scenes in the selected dataset")


if __name__ == "__main__":
    main()
