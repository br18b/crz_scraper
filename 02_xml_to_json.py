#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from lxml import etree
import xmljson

from crz_config import DUMP_DIR, OUT_JSON_DIR, BAD_XML_DIR

OUT_JSON_DIR.mkdir(exist_ok=True)
BAD_XML_DIR.mkdir(exist_ok=True)

parser = etree.XMLParser(
    recover=True,
    encoding="utf-8",
    huge_tree=True,
    resolve_entities=False,
    load_dtd=False,
    no_network=True,
)

# 1) Cleanup: stale temp files from previous crashes
for tmp in OUT_JSON_DIR.glob("*.json.tmp"):
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def atomic_write_json(obj: object, final_path: Path) -> None:
    """
    Write JSON atomically:
      - write to <final>.tmp
      - flush+fsync
      - os.replace(tmp, final)
    """
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    # If a stale tmp exists for some reason, remove it
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass

    # Write to tmp
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace
    os.replace(tmp_path, final_path)


for p in sorted(DUMP_DIR.glob("*.xml")):
    stem = p.stem
    out_path = OUT_JSON_DIR / f"{stem}.json"

    # 2) Restartability: skip if final JSON already exists
    # (Optional: consider size==0 as "bad" and reprocess)
    if out_path.exists() and out_path.stat().st_size > 0:
        continue

    try:
        xml_bytes = p.read_bytes()
        root = etree.fromstring(xml_bytes, parser)

        obj = xmljson.parker.data(root)
        atomic_write_json(obj, out_path)

        print(f"{p.name} -> {out_path.name}")

    except Exception as e:
        print(f"{p.name} FAILED: {e!r}")
        try:
            shutil.copy2(p, BAD_XML_DIR / p.name)
        except Exception:
            pass
        # If we failed mid-write and a tmp exists, clean it
        try:
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.unlink()
        except FileNotFoundError:
            pass
        continue
