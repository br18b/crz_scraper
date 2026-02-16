from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator
from collections.abc import Mapping, Iterable
import unicodedata

ONE_LETTER_WORDS = {"o", "a", "i", "u", "v", "z", "s", "k"}
KEEP_UPPER = {"SMS", "MMS", "GDPR", "SR", "EU", "IT", "IS", "API", "SK"}

TOK_RE = re.compile(r"[^\W\d_]{2,}|[^\W\d_]|[0-9]+|[^\s]", re.UNICODE)

def normalize_ascii_str(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", s.casefold())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_ascii(obj, *, normalize_keys: bool = False):
    if isinstance(obj, str):
        return normalize_ascii_str(obj)

    if isinstance(obj, Mapping):
        if normalize_keys:
            return {normalize_ascii(k, normalize_keys=True): normalize_ascii(v, normalize_keys=True)
                    for k, v in obj.items()}
        else:
            return {k: normalize_ascii(v, normalize_keys=False) for k, v in obj.items()}

    if isinstance(obj, list):
        return [normalize_ascii(v, normalize_keys=normalize_keys) for v in obj]

    if isinstance(obj, tuple):
        return tuple(normalize_ascii(v, normalize_keys=normalize_keys) for v in obj)

    if isinstance(obj, set):
        return {normalize_ascii(v, normalize_keys=normalize_keys) for v in obj}

    if isinstance(obj, frozenset):
        return frozenset(normalize_ascii(v, normalize_keys=normalize_keys) for v in obj)

    if isinstance(obj, Iterable):
        return [normalize_ascii(v, normalize_keys=normalize_keys) for v in obj]

    return obj

def _fix_joined_case(w: str) -> str:
    # DOHODA -> Dohoda (but keep acronyms)
    if w.isupper() and len(w) > 4 and w not in KEEP_UPPER:
        return w.capitalize()
    return w

def squash_spaced_letters(s: str, *, min_run: int = 3) -> str:
    """
    Joins spaced-out letter runs, but does NOT swallow real 1-letter Slovak words
    like 'o'/'a' that appear between runs:
      'P R Á V O a P O R I A D O K' -> 'PRÁVO a PORIADOK'
      'D O H O D A o ...' -> 'Dohoda o ...'
      'Z m l u v a o ...' -> 'Zmluva o ...'
    """
    s = s.replace("\u00A0", " ")  # NBSP -> space
    parts = TOK_RE.findall(s)

    def is_letter(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def is_word(tok: str) -> bool:
        return tok.isalpha() and len(tok) > 1

    out: list[str] = []
    i = 0

    while i < len(parts):
        tok = parts[i]

        if is_letter(tok):
            # collect a run of single-letter alphabetic tokens
            j = i
            run: list[str] = []
            while j < len(parts) and is_letter(parts[j]):
                run.append(parts[j])
                j += 1

            # process the run into segments, splitting on real 1-letter words inside it
            seg: list[str] = []

            def flush_seg():
                nonlocal seg
                if len(seg) >= min_run:
                    out.append(_fix_joined_case("".join(seg)))
                else:
                    out.extend(seg)
                seg = []

            for idx, ch in enumerate(run):
                # internal split: "... PRÁVO a PORIADOK ..."
                if (
                    ch.islower()
                    and ch in ONE_LETTER_WORDS
                    and seg
                    and idx + 1 < len(run)  # there are letters after it
                ):
                    # finish previous segment, emit the function word, start new segment
                    flush_seg()
                    out.append(ch)
                    continue

                seg.append(ch)

            # handle trailing "o/a/..." before a normal next word:
            # "... DOHODA o poskytovaní ..."
            next_tok = parts[j] if j < len(parts) else None
            if (
                seg
                and seg[-1].islower()
                and seg[-1] in ONE_LETTER_WORDS
                and next_tok is not None
                and is_word(next_tok)
                and len(seg) - 1 >= min_run
            ):
                out.append(_fix_joined_case("".join(seg[:-1])))
                out.append(seg[-1])
            else:
                flush_seg()

            i = j
            continue

        out.append(tok)
        i += 1

    # re-join with spaces, but avoid spaces before punctuation/brackets
    s2 = " ".join(out)
    s2 = re.sub(r"\s+([,.;:!?])", r"\1", s2)
    s2 = re.sub(r"\s+([)\]])", r"\1", s2)
    s2 = re.sub(r"([(\\[])\s+", r"\1", s2)
    return s2

@dataclass(frozen=True)
class ContractRef:
    file: str
    index: int


def _load_resolved_index(index_path: Path) -> dict[int, tuple[str, int]]:
    """
    index.json format:
      { "114737": {"file":"2011-01-14.json","index":0}, ... }
    returns: {114737: ("2011-01-14.json", 0), ...}
    """
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{index_path} is not a JSON object")

    out: dict[int, tuple[str, int]] = {}
    for k, v in raw.items():
        try:
            cid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        f = v.get("file")
        i = v.get("index")
        if isinstance(f, str):
            try:
                out[cid] = (f, int(i))
            except Exception:
                pass
    return out

def _load_unresolved_ids(unresolved_path: Path) -> set[int]:
    """
    unresolved.jsonl lines look like:
      {"id": 1091558, "reason": "...", ...}
    """
    out: set[int] = set()
    with open(unresolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = int(obj.get("id"))
                out.add(cid)
            except Exception:
                # tolerate partial / corrupted lines
                continue
    return out

def _records_list_from_dump_obj(obj: Any, *, verbose: bool, file_name: str) -> Optional[list[dict[str, Any]]]:
    if not obj or not isinstance(obj, dict):
        return None

    if "zmluva" not in obj:
        if verbose:
            print(f'{file_name} does not contain key "zmluva".')
        return None

    zmluvy = obj.get("zmluva")
    if zmluvy is None:
        if verbose:
            print(f"{file_name} has zmluva = null.")
        return None

    if isinstance(zmluvy, dict):
        if verbose:
            print(f"{file_name} has zmluva as dict - treating as single element list")
        return [zmluvy]

    if isinstance(zmluvy, list):
        return [r for r in zmluvy if isinstance(r, dict)]

    if verbose:
        print(f"{file_name} has zmluva but it's neither list nor dict, type={type(zmluvy).__name__}")
    return None


def _load_contract_by_ref(
    dump_dir: Path,
    ref: ContractRef,
    *,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load a single dump record by (file,index) where index is based on:
      - list zmluva: normal index
      - dict zmluva: index 0
    Returns the record dict or None.
    """
    p = dump_dir / ref.file
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        if verbose:
            print(f"READ FAIL: {p} {type(e).__name__}: {e}")
        return None

    recs = _records_list_from_dump_obj(obj, verbose=verbose, file_name=p.name)
    if not recs:
        return None

    if ref.index < 0 or ref.index >= len(recs):
        if verbose:
            print(f"Bad index for {p.name}: index={ref.index} n={len(recs)}")
        return None

    rec = recs[ref.index]
    if not isinstance(rec, dict):
        return None
    return rec


def iterate_contracts(
    dump_dir: Path,
    *,
    verbose: bool = True,
    yield_ref: bool = False,
    resolved_dir: Path | None = None,
    filtered_jsonl: Path | None = None,
) -> Iterator[Dict[str, Any]] | Iterator[tuple[ContractRef, Dict[str, Any]]]:
    # replay mode - ignore everything else
    if filtered_jsonl is not None:
        with open(filtered_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ref_obj = obj.get("ref") or {}
                file = ref_obj.get("file")
                idx = ref_obj.get("index")
                if not isinstance(file, str):
                    continue
                try:
                    index = int(idx)
                except Exception:
                    continue

                ref = ContractRef(file=file, index=index)
                rec = load_contract_by_ref(dump_dir, ref, verbose=verbose)
                if rec is None:
                    continue

                # sanity: if line has "id", ensure it matches record["ID"] (warn, don’t crash)
                try:
                    cid_line = int(obj.get("id"))
                    cid_rec = int(rec.get("ID"))
                    if cid_line != cid_rec and verbose:
                        print(f"[filtered_jsonl] ID mismatch: line id={cid_line} but record ID={cid_rec} ref={ref}")
                except Exception:
                    pass

                if yield_ref:
                    yield ref, rec
                else:
                    yield rec
        return

    # Optional: collision-aware filtering
    resolved_map: dict[int, tuple[str, int]] | None = None
    unresolved_ids: set[int] | None = None
    if resolved_dir is not None:
        index_path = resolved_dir / "index.json"
        unresolved_path = resolved_dir / "unresolved.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}")
        if not unresolved_path.exists():
            raise FileNotFoundError(f"Missing {unresolved_path}")

        resolved_map = _load_resolved_index(index_path)
        unresolved_ids = _load_unresolved_ids(unresolved_path)

        if verbose:
            print(
                f"[iterate_contracts] collisions enabled: "
                f"resolved={len(resolved_map)} unresolved={len(unresolved_ids)} from {resolved_dir}",
                flush=True,
            )

    for p in sorted(dump_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            if verbose:
                print("READ FAIL:", p, e)
            continue

        if not data or not isinstance(data, dict):
            if verbose and data and not isinstance(data, dict):
                print(f"{p} is not an object, type={type(data).__name__}")
            continue

        if "zmluva" not in data:
            if verbose:
                print(f'{p} does not contain key "zmluva".')
            continue

        zmluvy = data.get("zmluva")
        if zmluvy is None:
            if verbose:
                print(f"{p} has zmluva = null.")
            continue

        if isinstance(zmluvy, dict):
            if verbose:
                print(f"{p} has zmluva as dict - treating as single element list")
            zmluvy_list = [zmluvy]
        elif isinstance(zmluvy, list):
            zmluvy_list = zmluvy
        else:
            if verbose:
                print(f"{p} has zmluva but it's neither list nor dict, type={type(zmluvy).__name__}")
            continue

        for i, zmluva in enumerate(zmluvy_list):
            if not isinstance(zmluva, dict):
                if verbose:
                    print(f"{p}, element {i}: zmluva is not an object, type={type(zmluva).__name__}")
                continue

            # collision-aware skip logic (fast O(1))
            if resolved_map is not None and unresolved_ids is not None:
                raw_id = zmluva.get("ID")
                try:
                    cid = int(raw_id) if raw_id is not None else None
                except Exception:
                    cid = None

                if cid is not None:
                    if cid in unresolved_ids:
                        continue

                    chosen = resolved_map.get(cid)
                    if chosen is not None:
                        chosen_file, chosen_index = chosen
                        # if this isn't the chosen (file,index), skip it
                        if p.name != chosen_file or i != chosen_index:
                            continue

            # existing validation
            if "predmet" not in zmluva:
                if verbose:
                    print(f'{p}, element {i} does not contain "predmet".')
                continue

            predmet = zmluva.get("predmet")
            if predmet is None:
                continue
            if not isinstance(predmet, str):
                if verbose:
                    print(f'{p}, element {i} has a non-str "predmet"={predmet}, type={type(predmet).__name__}')
                continue

            if yield_ref:
                yield ContractRef(file=p.name, index=i), zmluva
            else:
                yield zmluva
