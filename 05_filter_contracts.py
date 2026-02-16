#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from crz_config import OUT_JSON_DIR, RESOLVED_COLLISIONS_DIR, FILTERED_CONTRACTS_DIR
from helper import iterate_contracts, squash_spaced_letters, normalize_ascii_str


# ---------------------
# filter config parsing
# ---------------------

_RX_WORDS_DEFAULT = r"[a-z0-9]+"

@dataclass(frozen=True)
class OrPattern:
    alts: Tuple[str, ...]

    def matches(self, tok: str, *, mode: str = "prefix") -> bool:
        # Special-case: "dat" should not match "datum"
        for a in self.alts:
            if not a:
                continue

            if a == "dat":
                if tok.startswith("datum"):
                    continue
                if tok.startswith("dat"):
                    return True
                continue

            if mode == "contains":
                if a in tok:
                    return True
            else:  # prefix
                if tok.startswith(a):
                    return True
        return False

def _parse_or_pattern(p: str) -> OrPattern:
    # normalize to ascii-lower
    p2 = normalize_ascii_str(p)
    alts = tuple(a.strip() for a in p2.split("|") if a.strip())
    return OrPattern(alts=alts)

@dataclass(frozen=True)
class SchemaNode:
    pat: OrPattern
    children: Tuple["SchemaNode", ...]

def _parse_schema_node(key_pat: str, val: Any) -> SchemaNode:
    pat = _parse_or_pattern(key_pat)

    # val can be:
    # - str: one child leaf
    # - dict: children mapping
    # - list: list of child specs (str or dict)
    children: List[SchemaNode] = []

    if isinstance(val, str):
        children.append(_parse_schema_node(val, {}))

    elif isinstance(val, dict):
        for k, v in val.items():
            children.append(_parse_schema_node(str(k), v))

    elif isinstance(val, list):
        for item in val:
            if isinstance(item, str):
                children.append(_parse_schema_node(item, {}))
            elif isinstance(item, dict):
                # allow {"poskyt": "udaj|dat"} as list element too
                for k, v in item.items():
                    children.append(_parse_schema_node(str(k), v))
            else:
                # ignore unknown items
                pass

    else:
        # unknown => leaf
        pass

    return SchemaNode(pat=pat, children=tuple(children))

def _parse_schema_root(schema_obj: Any) -> List[SchemaNode]:
    """
    Accepts either:
      schema: { "a|b": { "c": "d" } }
    or:
      schema: [ { ... }, { ... } ]
    """
    roots: List[SchemaNode] = []

    if isinstance(schema_obj, list):
        for it in schema_obj:
            roots.extend(_parse_schema_root(it))
        return roots

    if isinstance(schema_obj, dict):
        for k, v in schema_obj.items():
            roots.append(_parse_schema_node(str(k), v))
        return roots

    raise RuntimeError("schema must be a dict or list of dicts")

@dataclass(frozen=True)
class FilterConfig:
    token_re: re.Pattern
    pre_mode: str
    pre_include: Tuple[OrPattern, ...]
    pre_exclude: Tuple[OrPattern, ...]
    stanza_enabled: bool
    stanza_mode: str
    stanza_exclude: Tuple[OrPattern, ...]
    stanza_start_depth: int
    stanza_edge_scope: str
    stanza_schema_roots: Tuple[SchemaNode, ...]

def load_filter_config(path: Path) -> tuple[FilterConfig, str]:
    raw_text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    cfg = json.loads(raw_text)

    pre = cfg.get("precondition", {}) if isinstance(cfg, dict) else {}
    stanza = cfg.get("stanza", {}) if isinstance(cfg, dict) else {}

    token_re_s = pre.get("token_re", _RX_WORDS_DEFAULT)
    token_re = re.compile(token_re_s, re.IGNORECASE)

    pre_mode = str(pre.get("mode", "prefix")).strip().lower()
    if pre_mode not in ("prefix", "contains"):
        pre_mode = "prefix"

    pre_include = tuple(_parse_or_pattern(p) for p in (pre.get("include", []) or []))
    pre_exclude = tuple(_parse_or_pattern(p) for p in (pre.get("exclude", []) or []))

    stanza_enabled = bool(stanza.get("enabled", True))
    stanza_mode = str(stanza.get("mode", "prefix")).strip().lower()
    if stanza_mode not in ("prefix", "contains"):
        stanza_mode = "prefix"

    stanza_exclude = tuple(_parse_or_pattern(p) for p in (stanza.get("exclude", []) or []))
    stanza_start_depth = int(stanza.get("startDepth", 10**9))  # max depth allowed for schema root match

    stanza_edge_scope = str(stanza.get("edgeScope", "subtree")).strip().lower()
    if stanza_edge_scope not in ("subtree", "arg_or_conj"):
        stanza_edge_scope = "subtree"

    schema_obj = stanza.get("schema", {})
    stanza_schema_roots = tuple(_parse_schema_root(schema_obj))

    return (
        FilterConfig(
            token_re=token_re,
            pre_mode=pre_mode,
            pre_include=pre_include,
            pre_exclude=pre_exclude,
            stanza_enabled=stanza_enabled,
            stanza_mode=stanza_mode,
            stanza_exclude=stanza_exclude,
            stanza_start_depth=stanza_start_depth,
            stanza_edge_scope=stanza_edge_scope,
            stanza_schema_roots=stanza_schema_roots,
        ),
        sha,
    )


# -----------------
# stanza “big guns”
# -----------------

_NLP = None
_ARG_DEPRELS = {"nmod", "obj", "obl"}

def get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    import stanza
    stanza.download("sk", verbose=False)
    _NLP = stanza.Pipeline("sk", processors="tokenize,pos,lemma,depparse", verbose=False)
    return _NLP

def _norm_lemma(x: str) -> str:
    return normalize_ascii_str(x or "")

def _build_dep_index(words, *, skip_punct: bool = True):
    skip_upos = {"PUNCT"} if skip_punct else set()

    by_id = {w.id: w for w in words if getattr(w, "upos", None) not in skip_upos}

    children: dict[int, List[int]] = {i: [] for i in by_id.keys()}
    roots: List[int] = []

    for w in by_id.values():
        head = getattr(w, "head", 0)
        if head == 0 or head not in by_id:
            roots.append(w.id)
        else:
            children[head].append(w.id)

    for k in children:
        children[k].sort()
    roots.sort()

    return by_id, children, roots

def _compute_depths(roots: List[int], children: dict[int, List[int]]) -> dict[int, int]:
    depth: dict[int, int] = {}
    stack: List[tuple[int, int]] = [(r, 0) for r in roots]
    while stack:
        i, d = stack.pop()
        if i in depth:
            continue
        depth[i] = d
        for ch in children.get(i, []):
            stack.append((ch, d + 1))
    return depth

def _subtree_ids_cached(root: int, children: dict[int, List[int]], cache: dict[int, List[int]]) -> List[int]:
    if root in cache:
        return cache[root]
    out: List[int] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
        stack.extend(reversed(children.get(i, [])))
    cache[root] = out
    return out

def _any_excluded(lemmas: dict[int, str], exclude: Tuple[OrPattern, ...], *, mode: str) -> bool:
    if not exclude:
        return False
    for l in lemmas.values():
        for pat in exclude:
            if pat.matches(l, mode=mode):
                return True
    return False

def _conj_closure(start: int, children: dict[int, List[int]], by_id: dict[int, Any]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    stack = [start]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        for ch in children.get(x, []):
            dep = getattr(by_id[ch], "deprel", "") or ""
            if dep.split(":", 1)[0] == "conj":
                stack.append(ch)
    out.sort()
    return out

def _arg_or_conj_candidates(parent: int, children: dict[int, List[int]], by_id: dict[int, Any],
                            cache: dict[int, List[int]]) -> List[int]:
    if parent in cache:
        return cache[parent]

    out: List[int] = []
    seen: set[int] = set()

    for ch in children.get(parent, []):
        dep = getattr(by_id[ch], "deprel", "") or ""
        dep_base = dep.split(":", 1)[0]
        if dep_base in _ARG_DEPRELS:
            for x in _conj_closure(ch, children, by_id):
                if x not in seen:
                    seen.add(x)
                    out.append(x)

    out.sort()
    cache[parent] = out
    return out

def _match_schema_at(
    node_id: int,
    node: SchemaNode,
    lemmas: dict[int, str],
    children: dict[int, List[int]],
    by_id: dict[int, Any],
    subtree_cache: dict[int, List[int]],
    arg_cache: dict[int, List[int]],
    memo: dict[tuple[int, int], bool],
    *,
    mode: str,
    edge_scope: str,
) -> bool:
    key = (node_id, id(node))
    if key in memo:
        return memo[key]

    l = lemmas.get(node_id, "")
    if not node.pat.matches(l, mode=mode):
        memo[key] = False
        return False

    # choose candidate set for child-matching
    if edge_scope == "arg_or_conj":
        cands = _arg_or_conj_candidates(node_id, children, by_id, arg_cache)
    else:
        cands = _subtree_ids_cached(node_id, children, subtree_cache)
        cands = [x for x in cands if x != node_id]

    for ch_spec in node.children:
        ok_child = False
        for cand in cands:
            if _match_schema_at(
                cand, ch_spec, lemmas, children, by_id,
                subtree_cache, arg_cache, memo,
                mode=mode, edge_scope=edge_scope
            ):
                ok_child = True
                break
        if not ok_child:
            memo[key] = False
            return False

    memo[key] = True
    return True

def stanza_filter(title: str, nlp, cfg: FilterConfig) -> bool:
    doc = nlp(title)

    for sent in doc.sentences:
        words = sent.words
        if not words:
            continue

        by_id, children, roots = _build_dep_index(words, skip_punct=True)
        if not by_id:
            continue

        lemmas: dict[int, str] = {}
        for i, w in by_id.items():
            l = getattr(w, "lemma", None) or getattr(w, "text", "")
            lemmas[i] = _norm_lemma(l)

        # stanza-level exclude
        if _any_excluded(lemmas, cfg.stanza_exclude, mode=cfg.stanza_mode):
            continue

        depths = _compute_depths(roots, children)

        subtree_cache: dict[int, List[int]] = {}
        arg_cache: dict[int, List[int]] = {}
        memo: dict[tuple[int, int], bool] = {}

        # OR over schema roots
        for root_spec in cfg.stanza_schema_roots:
            for i in lemmas.keys():
                if depths.get(i, 10**9) > cfg.stanza_start_depth:
                    continue
                if _match_schema_at(
                    i, root_spec, lemmas, children, by_id,
                    subtree_cache, arg_cache, memo,
                    mode=cfg.stanza_mode,
                    edge_scope=cfg.stanza_edge_scope,     # <-- ADD
                ):
                    return True

    return False


# ------------------------------
# fast prefilter (config-driven)
# ------------------------------

def fast_prefilter(title: str, cfg: FilterConfig) -> bool:
    s2 = normalize_ascii_str(title)
    tokens = cfg.token_re.findall(s2)
    if not tokens:
        return False

    # exclude: if any token matches any exclude pattern => reject
    for tok in tokens:
        for pat in cfg.pre_exclude:
            if pat.matches(tok, mode=cfg.pre_mode):
                return False

    # include: must satisfy ALL include patterns
    for need in cfg.pre_include:
        if not any(need.matches(tok, mode=cfg.pre_mode) for tok in tokens):
            return False

    return True


# ------------------
# restartable output
# ------------------

@dataclass(frozen=True)
class RefKey:
    file: str
    index: int

    def as_tuple(self) -> tuple[str, int]:
        return (self.file, self.index)

def _parse_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def load_existing_ids(jsonl_path: Path) -> set[int]:
    seen: set[int] = set()
    if not jsonl_path.exists():
        return seen
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = _parse_int(obj.get("id"))
                if cid is not None:
                    seen.add(cid)
            except Exception:
                continue
    return seen

def load_checkpoint(checkpoint_path: Path) -> Optional[RefKey]:
    if not checkpoint_path.exists():
        return None
    try:
        obj = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    f = obj.get("file")
    i = obj.get("index")
    if not isinstance(f, str):
        return None
    ii = _parse_int(i)
    if ii is None:
        return None
    return RefKey(file=f, index=ii)

def save_checkpoint(checkpoint_path: Path, ref: RefKey, *, found: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"file": ref.file, "index": ref.index, "found": found, "ts": time.time()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, checkpoint_path)

def append_jsonl_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


# -------------
# record filter
# -------------

def filter_rec(rec: Dict[str, Any], *, nlp, cfg: FilterConfig) -> bool:
    predmet = rec["predmet"]  # guaranteed str by iterate_contracts
    title = squash_spaced_letters(predmet)

    if not fast_prefilter(title, cfg):
        return False

    if not cfg.stanza_enabled:
        return True  # prefilter-only mode

    return stanza_filter(title, nlp, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", type=str, default="", help="Path to filter.json (default: next to this script).")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N records (0 = no limit).")
    ap.add_argument("--fresh", action="store_true", help="Delete outputs and start from scratch.")
    ap.add_argument("--checkpoint-seconds", type=float, default=20.0, help="How often to write checkpoint.")
    ap.add_argument("--verbose", action="store_true", help="Print progress info.")
    args = ap.parse_args()

    out_dir = OUT_JSON_DIR
    resolved_dir = RESOLVED_COLLISIONS_DIR

    # 1) Resolve filter path + load it -> gives cfg + sha
    filter_path = Path(args.filter) if args.filter else Path(__file__).with_name("filter.json")
    cfg, filter_sha = load_filter_config(filter_path)

    # 2) Create per-filter output directory
    base_root = Path(FILTERED_CONTRACTS_DIR)
    base_root.mkdir(parents=True, exist_ok=True)

    run_root = base_root / filter_sha
    run_root.mkdir(parents=True, exist_ok=True)

    out_jsonl = run_root / "filtered.jsonl"
    checkpoint_path = run_root / "_checkpoint.json"

    # 3) Optional: snapshot the filter used (best-effort)
    try:
        (run_root / "filter.json").write_text(filter_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    if args.fresh:
        for p in (out_jsonl, checkpoint_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    seen_ids = load_existing_ids(out_jsonl)
    resume_from = load_checkpoint(checkpoint_path)

    if args.verbose:
        print(f"[filter] path={filter_path} sha256={filter_sha}", flush=True)
        print(f"[resume] jsonl={out_jsonl.exists()} seen_ids={len(seen_ids)}", flush=True)
        print(f"[resume] checkpoint={checkpoint_path.exists()} at={resume_from.as_tuple() if resume_from else None}", flush=True)

    nlp = get_nlp()

    it = iterate_contracts(out_dir, verbose=False, resolved_dir=resolved_dir, yield_ref=True)
    pbar = tqdm(it, desc="Filtering contracts", unit="rec")

    t_last_ckpt = time.perf_counter()
    last_ref: Optional[RefKey] = None
    found = 0

    with open(out_jsonl, "a", encoding="utf-8") as fj:
        for k, (ref, rec) in enumerate(pbar, start=1):
            rk = RefKey(file=ref.file, index=ref.index)
            last_ref = rk

            # resume skip
            if resume_from is not None and rk.as_tuple() <= resume_from.as_tuple():
                continue

            cid = _parse_int(rec.get("ID"))
            if cid is None:
                continue
            if cid in seen_ids:
                continue

            if filter_rec(rec, nlp=nlp, cfg=cfg):
                append_jsonl_line(
                    fj,
                    {
                        "id": cid,
                        "predmet": rec["predmet"],
                        "ref": {"file": ref.file, "index": ref.index},
                    },
                )
                seen_ids.add(cid)
                found += 1
                pbar.set_postfix(found=found)

            now = time.perf_counter()
            if args.checkpoint_seconds > 0 and (now - t_last_ckpt) >= float(args.checkpoint_seconds):
                if last_ref is not None:
                    save_checkpoint(checkpoint_path, last_ref, found=found)
                t_last_ckpt = now

            if args.limit and k >= args.limit:
                break

    pbar.close()

    if last_ref is not None:
        save_checkpoint(checkpoint_path, last_ref, found=found)

    if args.verbose:
        print(f"[out] appended {found} new matches -> {out_jsonl}", flush=True)
        print(f"[out] checkpoint -> {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
