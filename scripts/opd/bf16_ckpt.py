#!/usr/bin/env python
"""Cast verl-exported HuggingFace checkpoints to BF16 in place.

verl's FSDPCheckpointManager builds the HF model shell with
``torch_dtype=torch.bfloat16`` but then calls ``save_pretrained(state_dict=...)``
with the actor's FP32 state dict, so the exported safetensors end up FP32
(2x the size). This utility rewrites each shard as BF16, updates config.json
and the safetensors index metadata, and drops a ``.bf16_converted`` marker.

Usage:
  # convert one checkpoint directory (the .../huggingface dir)
  python bf16_ckpt.py --once <run_dir>/global_step_10/actor/huggingface

  # watch a run directory and convert every new checkpoint as it appears
  python bf16_ckpt.py --watch <run_dir>
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MARKER = ".bf16_converted"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def read_shard_dtypes(path: str) -> set[str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {v["dtype"] for k, v in header.items() if k != "__metadata__"}


def convert_shard(path: str, stage_dir: str | None = None) -> bool:
    """Rewrite one safetensors shard as BF16. Returns True if it changed.

    With ``stage_dir`` on a different filesystem, the BF16 temp is written
    there first, then the FP32 shard is deleted before copying the BF16 file
    back. Peak usage on the checkpoint filesystem is therefore one shard
    (instead of FP32 + BF16 temp), which is what saves a nearly full disk.
    """
    dtypes = read_shard_dtypes(path)
    if "F32" not in dtypes:
        return False
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
        tmp = os.path.join(stage_dir, os.path.basename(path) + ".bf16.tmp")
    else:
        tmp = path + ".bf16.tmp"
    tensors = {}
    metadata = None
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            t = f.get_tensor(key)
            tensors[key] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
    save_file(tensors, tmp, metadata=metadata)
    del tensors
    if os.path.dirname(os.path.abspath(tmp)) == os.path.dirname(os.path.abspath(path)):
        os.replace(tmp, path)
    else:
        # Cross-device: free the FP32 shard first, then copy the BF16 back.
        os.unlink(path)
        shutil.copyfile(tmp, path)
        os.unlink(tmp)
    return True


def convert_checkpoint(hf_dir: str, stage_dir: str | None = None) -> bool:
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(hf_dir, "*.safetensors")))
    # Convert smallest shards first: even without a stage dir this lowers the
    # peak temp requirement for the largest shard.
    shards = sorted(shards, key=lambda s: os.path.getsize(os.path.join(hf_dir, s)))

    changed = False
    for shard in shards:
        path = os.path.join(hf_dir, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        before = read_shard_dtypes(path)
        if "F32" in before:
            log(f"converting {shard}: {sorted(before)} -> BF16")
            convert_shard(path, stage_dir)
            changed = True
        after = read_shard_dtypes(path)
        if "F32" in after:
            raise RuntimeError(f"{path} still contains F32 after conversion: {after}")
        log(f"{shard}: dtypes={sorted(after)} size={os.path.getsize(path)/2**30:.2f} GiB")

    # Mark the config as BF16 so from_pretrained(default) does not upcast.
    config_path = os.path.join(hf_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["torch_dtype"] = "bfloat16"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    # Refresh total_size in the safetensors index.
    if os.path.isfile(index_path):
        total = sum(os.path.getsize(os.path.join(hf_dir, s)) for s in shards)
        with open(index_path) as f:
            index = json.load(f)
        index.setdefault("metadata", {})["total_size"] = total
        with open(index_path, "w") as f:
            json.dump(index, f)
            f.write("\n")

    marker = os.path.join(hf_dir, MARKER)
    with open(marker, "w") as f:
        json.dump({"converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "shards": shards}, f)
        f.write("\n")
    log(f"done: {hf_dir} (changed={changed})")
    return changed


def shard_snapshot(hf_dir: str):
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    snap = []
    for s in shards:
        p = os.path.join(hf_dir, s)
        if not os.path.isfile(p):
            return None
        snap.append((s, os.path.getsize(p)))
    return tuple(snap)


def watch(run_dir: str, poll_s: int = 15, quiet_s: int = 30, stage_dir: str | None = None) -> None:
    log(f"watching {run_dir} for HF checkpoints (stage_dir={stage_dir})")
    last_snap: dict[str, tuple] = {}
    stable_since: dict[str, float] = {}
    seen: set[str] = set()
    while True:
        pattern = os.path.join(run_dir, "global_step_*", "actor", "huggingface")
        for hf_dir in sorted(glob.glob(pattern), key=lambda p: int(p.split("global_step_")[1].split("/")[0])):
            if os.path.isfile(os.path.join(hf_dir, MARKER)):
                seen.add(hf_dir)
                continue
            snap = shard_snapshot(hf_dir)
            if snap is None:
                continue
            now = time.time()
            if last_snap.get(hf_dir) != snap:
                last_snap[hf_dir] = snap
                stable_since[hf_dir] = now
                continue
            if now - stable_since.get(hf_dir, now) < quiet_s:
                continue
            log(f"detected completed checkpoint: {hf_dir}")
            try:
                convert_checkpoint(hf_dir, stage_dir)
            except Exception as exc:  # keep the daemon alive; retry next round
                log(f"ERROR converting {hf_dir}: {exc!r}")
                continue
            seen.add(hf_dir)
            stable_since.pop(hf_dir, None)
            last_snap.pop(hf_dir, None)
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", help="path to one .../actor/huggingface directory")
    g.add_argument("--watch", help="run directory containing global_step_*")
    parser.add_argument("--poll", type=int, default=15)
    parser.add_argument("--quiet", type=int, default=30)
    parser.add_argument(
        "--stage-dir",
        default=None,
        help="write BF16 temp files here (use a dir on another filesystem when the checkpoint disk is nearly full)",
    )
    args = parser.parse_args()
    if args.once:
        convert_checkpoint(args.once, args.stage_dir)
    else:
        watch(args.watch, poll_s=args.poll, quiet_s=args.quiet, stage_dir=args.stage_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
