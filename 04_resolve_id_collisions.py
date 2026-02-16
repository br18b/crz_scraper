#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from crz_config import OUT_JSON_DIR, RESOLVED_COLLISIONS_DIR
from crz_page import fetch_and_parse_crz_id
from helper import iterate_contracts

_RX_DIGITS = re.compile(r"\d+")
_RX_DATE_PREFIX = re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})")


def norm_ico(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    digits = "".join(_RX_DIGITS.findall(s))
    if not digits:
        return None
    digits = digits.lstrip("0")
    return digits if digits else None


def norm_predmet(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).replace("\u00A0", " ")
    s = " ".join(s.split()).strip()
    return s if s else None


def file_date_str(file: str) -> str:
    m = _RX_DATE_PREFIX.match(file)
    return m.group("d") if m else ""


@dataclass(frozen=True)
class Ref:
    file: str
    index: int


def get_key_or_prefixed(d: dict[str, Any], key: str) -> tuple[Any, bool]:
    if key in d:
        return d[key], True
    pref = key + "_"
    for k in d.keys():
        if k.startswith(pref):
            return d[k], True
    return None, False


def extract_authoritative(page: dict[str, Any]) -> dict[str, Any]:
    ident_ascii = page.get("identifikacia_ascii") or {}

    nazov_raw, nazov_present = get_key_or_prefixed(ident_ascii, "nazov_zmluvy")
    if not nazov_present:
        nazov_raw, nazov_present = get_key_or_prefixed(ident_ascii, "nazov")
    if not nazov_present:
        nazov_raw, nazov_present = get_key_or_prefixed(ident_ascii, "predmet")

    ico_raw, ico_present = get_key_or_prefixed(ident_ascii, "ico_dodavatela")
    ico1_raw, ico1_present = get_key_or_prefixed(ident_ascii, "ico_objednavatela")

    return {
        "predmet": norm_predmet(nazov_raw),
        "predmet_present": nazov_present,
        "ico": norm_ico(ico_raw),
        "ico_present": ico_present,
        "ico1": norm_ico(ico1_raw),
        "ico1_present": ico1_present,
    }


def matches_authoritative(auth: dict[str, Any], rec: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    r_pred = norm_predmet(rec.get("predmet"))
    r_ico = norm_ico(rec.get("ico"))
    r_ico1 = norm_ico(rec.get("ico1"))

    def check(field: str, r_val: Any) -> bool:
        present = bool(auth.get(f"{field}_present"))
        a_val = auth.get(field)

        if not present:
            reasons.append(f"{field}: not on page (ignored)")
            return True

        if a_val is None:
            ok = (r_val is None)
            reasons.append(f"{field}: page blank => record {'OK' if ok else 'NOT NULL'}")
            return ok

        ok = (r_val == a_val)
        reasons.append(f"{field}: {'matches' if ok else 'differs'}")
        return ok

    ok = True
    ok &= check("ico", r_ico)
    ok &= check("ico1", r_ico1)
    ok &= check("predmet", r_pred)
    return ok, reasons


def domain_key(rec: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (norm_ico(rec.get("ico")), norm_ico(rec.get("ico1")), norm_predmet(rec.get("predmet")))


def choose_latest(refs: list[Ref]) -> Ref:
    # latest file date, then highest index, then filename
    return max(refs, key=lambda r: (file_date_str(r.file), r.index, r.file))


def _unlink_if_exists(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def load_already_resolved(resolved_path: Path) -> tuple[dict[str, Any], set[int]]:
    """
    If resolved.jsonl exists, load it and return:
      - resolved_index: dict[str(id)] -> {"file":..., "index":...}
      - done_ids: set[int]
    """
    resolved_index: dict[str, Any] = {}
    done: set[int] = set()
    if not resolved_path.exists():
        return resolved_index, done

    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = int(obj.get("id"))
                chosen = obj.get("chosen")
                if isinstance(chosen, dict) and "file" in chosen and "index" in chosen:
                    resolved_index[str(cid)] = {"file": chosen["file"], "index": int(chosen["index"])}
                    done.add(cid)
            except Exception:
                # ignore corrupted lines / partial writes
                continue

    return resolved_index, done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="Delete outputs and start from scratch (no resume).")
    ap.add_argument("--checkpoint-seconds", type=float, default=10.0, help="How often to write checkpoint index.")
    ap.add_argument("--flush-every", type=int, default=200, help="Flush JSONL outputs every N processed ids.")
    args = ap.parse_args()

    out_dir = OUT_JSON_DIR
    collisions_path = out_dir.parent / "_id_collisions.jsonl"

    resolved_dir = RESOLVED_COLLISIONS_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = resolved_dir / "resolved.jsonl"
    unresolved_path = resolved_dir / "unresolved.jsonl"
    index_path = resolved_dir / "index.json"
    checkpoint_path = resolved_dir / "_checkpoint_index.json"

    t0 = time.perf_counter()
    print(f"[start] out_dir={out_dir}", flush=True)
    print(f"[start] collisions_path={collisions_path} exists={collisions_path.exists()}", flush=True)

    if args.fresh:
        print("[start] --fresh: deleting previous outputs", flush=True)
        _unlink_if_exists(resolved_path)
        _unlink_if_exists(unresolved_path)
        _unlink_if_exists(index_path)
        _unlink_if_exists(checkpoint_path)

    # 1) Load collisions
    t_load0 = time.perf_counter()
    collisions: list[dict[str, Any]] = []
    with open(collisions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                collisions.append(json.loads(line))
    t_load1 = time.perf_counter()
    print(f"[load] collisions={len(collisions)}  dt={t_load1 - t_load0:.2f}s", flush=True)
    if not collisions:
        print("[load] no collisions -> nothing to do", flush=True)
        return

    # 2) Build wanted ids and fast ref lookup sets
    wanted: set[int] = set()
    refset_by_id: dict[int, set[tuple[str, int]]] = {}

    for row in collisions:
        cid = int(row["id"])
        wanted.add(cid)
        refs = [Ref(file=r[0], index=int(r[1])) for r in row["refs"]]
        refset_by_id[cid] = {(rr.file, rr.index) for rr in refs}

    print(f"[prep] wanted ids={len(wanted)}", flush=True)

    # 2.5) Resume: load already resolved ids from resolved.jsonl
    resolved_index, done_ids = load_already_resolved(resolved_path) if not args.fresh else ({}, set())
    if done_ids:
        print(f"[resume] already resolved ids loaded from {resolved_path.name}: {len(done_ids)}", flush=True)

    # 3) Gather pass: scan parsed contracts ONCE and gather candidates only for wanted ids
    t_gather0 = time.perf_counter()
    cands_by_id: dict[int, list[tuple[Ref, dict[str, Any]]]] = {cid: [] for cid in wanted}

    scanned = 0
    hits = 0
    hit_ids: set[int] = set()

    scan_bar = tqdm(
        desc="Gather: scanning parsed contracts",
        unit="rec",
        mininterval=0.5,
        smoothing=0.05,
    )

    last_postfix = time.perf_counter()
    for ref, rec in iterate_contracts(out_dir, verbose=False, yield_ref=True):
        scanned += 1
        scan_bar.update(1)

        raw = rec.get("ID")
        if raw is None:
            continue
        try:
            cid = int(raw)
        except Exception:
            continue
        if cid not in wanted:
            continue

        # keep only exact refs from collisions list
        if (ref.file, int(ref.index)) in refset_by_id[cid]:
            cands_by_id[cid].append((Ref(ref.file, int(ref.index)), rec))
            hits += 1
            hit_ids.add(cid)

        now = time.perf_counter()
        if now - last_postfix > 1.0:
            scan_bar.set_postfix(scanned=scanned, hits=hits, hit_ids=len(hit_ids))
            last_postfix = now

    scan_bar.close()
    t_gather1 = time.perf_counter()

    have_any = sum(1 for cid in wanted if cands_by_id.get(cid))
    print(
        f"[gather] scanned={scanned} hits={hits} hit_ids={len(hit_ids)} ids_with_candidates={have_any}/{len(wanted)}  dt={t_gather1 - t_gather0:.2f}s",
        flush=True,
    )

    # 4) Resolve pass: decide each collided id, writing continuously
    n_equiv = 0
    n_web = 0
    n_unresolved = 0
    fetch_time_total = 0.0

    session = requests.Session()

    t_resolve0 = time.perf_counter()

    # We write resolved/unresolved incrementally (no per-id files, no reopen per line)
    # resolved.jsonl contains only decisions that actually chose a ref
    # unresolved.jsonl contains diagnostics
    to_process = sorted(wanted - done_ids)
    print(f"[resolve] to_process={len(to_process)} (skipping already_done={len(done_ids)})", flush=True)

    resolve_bar = tqdm(
        total=len(to_process),
        desc="Resolve: deciding per collided id",
        unit="id",
        mininterval=0.5,
        smoothing=0.05,
    )

    last_checkpoint = time.perf_counter()

    with open(resolved_path, "a", encoding="utf-8") as fr, open(unresolved_path, "a", encoding="utf-8") as fu:
        for idx, cid in enumerate(to_process, start=1):
            resolve_bar.update(1)

            cands = cands_by_id.get(cid) or []
            if not cands:
                fu.write(json.dumps({"id": cid, "reason": "no candidates gathered"}, ensure_ascii=False) + "\n")
                n_unresolved += 1
                continue

            # If domain-equivalent => resolve immediately (no web)
            keys = {domain_key(rec) for _, rec in cands}
            if len(keys) == 1:
                chosen_ref = choose_latest([r for r, _ in cands])
                chosen = {"file": chosen_ref.file, "index": chosen_ref.index}
                resolved_index[str(cid)] = chosen
                fr.write(json.dumps({"id": cid, "status": "equiv", "chosen": chosen}, ensure_ascii=False) + "\n")
                n_equiv += 1
            else:
                # Otherwise fetch authoritative page and match
                n_web += 1
                t_web0 = time.perf_counter()
                try:
                    page = fetch_and_parse_crz_id(cid, sleep_s=0.0, session=session, min_interval_s=0.5)
                except Exception as e:
                    fu.write(
                        json.dumps({"id": cid, "reason": f"fetch failed: {type(e).__name__}: {e}"}, ensure_ascii=False) + "\n"
                    )
                    n_unresolved += 1
                    continue
                t_web1 = time.perf_counter()
                fetch_time_total += (t_web1 - t_web0)

                if not page.get("found", False):
                    fu.write(json.dumps({"id": cid, "reason": "page not found", "page": page}, ensure_ascii=False) + "\n")
                    n_unresolved += 1
                    continue

                auth = extract_authoritative(page)

                matches: list[Ref] = []
                for r, rec in cands:
                    ok, _reasons = matches_authoritative(auth, rec)
                    if ok:
                        matches.append(r)

                if len(matches) == 1:
                    chosen_ref = matches[0]
                    chosen = {"file": chosen_ref.file, "index": chosen_ref.index}
                    resolved_index[str(cid)] = chosen
                    fr.write(json.dumps({"id": cid, "status": "web_unique", "chosen": chosen}, ensure_ascii=False) + "\n")
                elif len(matches) >= 2:
                    chosen_ref = choose_latest(matches)
                    chosen = {"file": chosen_ref.file, "index": chosen_ref.index}
                    resolved_index[str(cid)] = chosen
                    fr.write(json.dumps({"id": cid, "status": "web_multi", "chosen": chosen, "n_matches": len(matches)}, ensure_ascii=False) + "\n")
                else:
                    fu.write(
                        json.dumps(
                            {
                                "id": cid,
                                "reason": "no candidate matches authoritative page",
                                "authoritative": auth,
                                "candidates": [
                                    {"ref": {"file": r.file, "index": r.index}, "key": domain_key(rec)} for r, rec in cands
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_unresolved += 1

            # tqdm postfix (not every iter)
            if idx % 100 == 0:
                avg_fetch = (fetch_time_total / n_web) if n_web else 0.0
                resolve_bar.set_postfix(res=len(resolved_index), equiv=n_equiv, web=n_web, unr=n_unresolved, avg_fetch=f"{avg_fetch:.2f}s")

            # Flush occasionally so we see updates on disk
            if args.flush_every > 0 and (idx % args.flush_every == 0):
                fr.flush()
                fu.flush()

            # checkpoint every N seconds (index snapshot)
            now = time.perf_counter()
            if now - last_checkpoint >= float(args.checkpoint_seconds):
                checkpoint_path.write_text(json.dumps(resolved_index, ensure_ascii=False, indent=2), encoding="utf-8")
                last_checkpoint = now

        fr.flush()
        fu.flush()

    resolve_bar.close()
    t_resolve1 = time.perf_counter()

    # final index.json
    index_path.write_text(json.dumps(resolved_index, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.", flush=True)
    print(f"[time] total={time.perf_counter()-t0:.2f}s", flush=True)
    print(f"[time] gather={t_gather1 - t_gather0:.2f}s  resolve={t_resolve1 - t_resolve0:.2f}s", flush=True)
    if n_web:
        print(f"[web] fetches={n_web} avg_fetch+parse={fetch_time_total/n_web:.3f}s", flush=True)
    print(f"[result] resolved={len(resolved_index)} / wanted={len(wanted)}  equiv={n_equiv}  web={n_web}  unresolved={n_unresolved}", flush=True)
    print(f"[out] resolved_jsonl={resolved_path}", flush=True)
    print(f"[out] unresolved_jsonl={unresolved_path}", flush=True)
    print(f"[out] index={index_path}", flush=True)
    print(f"[out] checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
