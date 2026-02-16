#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

from crz_config import OUT_JSON_DIR
from helper import iterate_contracts


def main() -> None:
    out_dir = OUT_JSON_DIR
    collisions_path = out_dir.parent / "_id_collisions.jsonl"

    # start fresh each run
    collisions_path.unlink(missing_ok=True)

    first: dict[int, object] = {}          # id -> ContractRef (from helper)
    dups: dict[int, list[list[object]]] = {}  # id -> [[file, index], [file, index], ...]
    scanned = 0
    bad_id = 0

    t0 = time.time()
    last = 0.0

    for ref, rec in iterate_contracts(out_dir, verbose=False, yield_ref=True):
        scanned += 1

        raw = rec.get("ID")
        if raw is None:
            bad_id += 1
            continue
        try:
            cid = int(raw)
        except Exception:
            bad_id += 1
            continue

        prev_ref = first.get(cid)
        if prev_ref is None:
            first[cid] = ref
        else:
            # first time we detect a dup, seed list with the original ref
            if cid not in dups:
                dups[cid] = [[prev_ref.file, prev_ref.index]]
            dups[cid].append([ref.file, ref.index])

        now = time.time()
        if now - last > 0.25:
            rate = scanned / (now - t0) if now > t0 else 0.0
            print(
                f"\rscanned={scanned}  unique={len(first)}  dup_ids={len(dups)}  bad_id={bad_id}  rate={rate:.1f}/s",
                end="",
                flush=True,
            )
            last = now

    # write one JSONL row per duplicated ID
    with open(collisions_path, "w", encoding="utf-8") as f:
        for cid in sorted(dups):
            row = {"id": cid, "refs": dups[cid]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Scanned records:   {scanned}")
    print(f"Unique IDs:        {len(first)}")
    print(f"IDs with dups:     {len(dups)}")
    print(f"Bad/missing IDs:   {bad_id}")
    print(f"Collisions log:    {collisions_path}")


if __name__ == "__main__":
    main()
