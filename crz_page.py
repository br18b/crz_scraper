from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, unquote

import unicodedata
import requests
from lxml import html

from time import perf_counter

from crz_config import PAGES_DIR

_LAST_NET_TS: float | None = None

def throttle_before_network(min_interval_s: float) -> None:
    """
    Ensures at least min_interval_s elapsed between *network* requests.
    No-op for min_interval_s <= 0.
    """
    global _LAST_NET_TS
    if min_interval_s <= 0:
        return

    now = perf_counter()
    if _LAST_NET_TS is not None:
        dt = now - _LAST_NET_TS
        if dt < min_interval_s:
            time.sleep(min_interval_s - dt)

def mark_network_request() -> None:
    global _LAST_NET_TS
    _LAST_NET_TS = perf_counter()

_RX_SAFE_FN = re.compile(r"[^a-zA-Z0-9._-]+")
_RX_MULTI_US = re.compile(r"_+")

def cache_path_from_url(url: str) -> Path:
    """
    https://www.crz.gov.sk/115902              -> PAGES_DIR/115902.html
    https://www.crz.gov.sk/zmluva/11992976/    -> PAGES_DIR/zmluva_11992976.html

    - strips scheme+domain
    - uses ONLY path (drops ?query and #fragment)
    - '/' -> '_'
    - unsafe chars -> '_'
    """
    u = urlparse(url)
    path = (u.path or "").strip("/")  # domain stripped automatically
    if not path:
        path = "root"

    name = path.replace("/", "_")
    name = _RX_SAFE_FN.sub("_", name)
    name = _RX_MULTI_US.sub("_", name).strip("_")

    # optional: keep filenames sane
    if len(name) > 200:
        name = name[:200].rstrip("_")

    return Path(PAGES_DIR) / f"{name}.html"

def fetch(
    url: str,
    *,
    timeout=(10, 60),
    session: Optional[requests.Session] = None,
    min_interval_s: float = 0.0,
    refetch_days: float = 7.0,
) -> FetchResult:
    # cache hit?
    p: Path | None = None
    try:
        p = cache_path_from_url(url)
        if p.exists():
            # if file is "fresh enough", use it
            if refetch_days is not None and refetch_days > 0:
                age_s = time.time() - p.stat().st_mtime
                if age_s < refetch_days * 86400.0: # 24 * 60 * 60
                    html_text = p.read_text(encoding="utf-8")
                    return FetchResult(attempt_url=url, final_url=url, status_code=200, html_text=html_text)
            else:
                # refetch_days <= 0 => never refetch, always use cache if present
                html_text = p.read_text(encoding="utf-8")
                return FetchResult(attempt_url=url, final_url=url, status_code=200, html_text=html_text)
    except Exception:
        # caching should never break scraping
        p = None

    # network fetch (rate-limited)
    throttle_before_network(min_interval_s)

    s = session or requests
    r = s.get(
        url,
        headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
        timeout=timeout,
        allow_redirects=True,
    )
    mark_network_request()

    r.encoding = "utf-8"
    r.raise_for_status()

    fr = FetchResult(attempt_url=url, final_url=r.url, status_code=r.status_code, html_text=r.text)

    # save cache best-effort (atomic)
    try:
        if p is None:
            p = cache_path_from_url(url)
        atomic_write_text(p, fr.html_text, encoding="utf-8")
    except Exception:
        pass

    return fr


def fetch_crz_id_pages(
    crz_id: int,
    *,
    sleep_s: float = 0.0,
    session: Optional[requests.Session] = None,
    min_interval_s: float = 0.0,
    refetch_days: float = 7.0,
) -> list[FetchResult]:
    out: list[FetchResult] = []
    for fmt in URL_VARIANTS:
        u = fmt.format(id=crz_id)
        try:
            out.append(fetch(u, session=session, min_interval_s=min_interval_s, refetch_days=refetch_days))
        except requests.HTTPError:
            pass
        except requests.RequestException:
            pass

        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


def fetch_and_parse_crz_url(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    min_interval_s: float = 0.0,
    refetch_days: float = 7.0,
) -> dict[str, Any]:
    try:
        fr = fetch(url, session=session, min_interval_s=min_interval_s, refetch_days=refetch_days)
    except requests.HTTPError:
        return {"url": url, "found": False, "warnings": [f"HTTP error for url={url}"]}
    except requests.RequestException as e:
        return {"url": url, "found": False, "warnings": [f"Request error for url={url}: {e}"]}

    p = parse_crz_page(fr.final_url, fr.html_text)
    p["found"] = True
    p["fetched_from"] = [{"attempt_url": url, "final_url": fr.final_url, "status": fr.status_code}]
    return p


def fetch_and_parse_crz_id(
    crz_id: int,
    *,
    sleep_s: float = 0.0,
    session: Optional[requests.Session] = None,
    min_interval_s: float = 0.0,
    refetch_days: float = 7.0,
) -> dict[str, Any]:
    frs = fetch_crz_id_pages(
        crz_id,
        sleep_s=sleep_s,
        session=session,
        min_interval_s=min_interval_s,
        refetch_days=refetch_days,
    )
    if not frs:
        return {"id": crz_id, "found": False, "warnings": [f"no working URL variant for id={crz_id}"]}

    parsed = []
    for fr in frs:
        p = parse_crz_page(fr.final_url, fr.html_text)
        parsed.append((fr, p))

    if len(parsed) >= 2:
        def core(p: dict[str, Any]) -> tuple[Any, Any, Any]:
            ident = p.get("identifikacia", {})
            num = ident.get("Č. zmluvy") or ident.get("Číslo dodatku") or ident.get("Č. dodatku")
            return (p.get("page_id"), p.get("title"), num)

        c0 = core(parsed[0][1])
        for _, pp in parsed[1:]:
            if core(pp) != c0:
                stink(
                    "Both URL variants worked but parsed identity differs:\n"
                    f"  variant1={parsed[0][0].attempt_url} -> {parsed[0][0].final_url} core={c0}\n"
                    f"  variant2={parsed[1][0].attempt_url} -> {parsed[1][0].final_url} core={core(pp)}\n"
                )

    fr0, p0 = parsed[0]
    p0["id"] = crz_id
    p0["found"] = True
    p0["fetched_from"] = [
        {"attempt_url": fr.attempt_url, "final_url": fr.final_url, "status": fr.status_code}
        for fr, _ in parsed
    ]
    return p0

# ---------- basic “raise stink” ----------
class StinkError(RuntimeError):
    pass


def stink(msg: str) -> None:
    raise StinkError(msg)


# ---------- HTTP ----------
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"

CRZ_BASE = "https://www.crz.gov.sk/"
URL_VARIANTS = [
    "https://www.crz.gov.sk/{id}",
    "https://www.crz.gov.sk/zmluva/{id}",
]


@dataclass
class FetchResult:
    attempt_url: str
    final_url: str
    status_code: int
    html_text: str

def strip_diacritics(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def ascii_key(label: str) -> str:
    """
    Convert 'IČO objednávateľa' -> 'ico_objednavatela', etc.
    Stable keys for downstream processing.
    """
    s = strip_diacritics(label)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")

    # nice-to-have: a couple common upgrades
    if s in ("c_zmluvy", "c_zmluvy_"):
        s = "cislo_zmluvy"
    return s


def dict_ascii_mirror(src: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in src.items():
        ak = ascii_key(k)
        if ak in out:
            i = 2
            while f"{ak}_{i}" in out:
                i += 1
            ak = f"{ak}_{i}"
        out[ak] = v
    return out



# ---------- parsing helpers ----------
_WS = re.compile(r"\s+")
_MONEY = re.compile(r"([-+]?\d[\d\s\u00a0]*)(?:,(\d+))?")  # "12 345,67"


def norm_space(s: str) -> str:
    return _WS.sub(" ", s).strip()


def node_text_lines(node) -> list[str]:
    # Preserve <br> as line breaks: use itertext and split on explicit breaks
    # lxml: <br> yields no text, so we do manual handling
    parts: list[str] = []
    for t in node.itertext():
        tt = norm_space(t)
        if tt:
            parts.append(tt)
    # This loses explicit line breaks, but keeps address chunks; good enough for compare/store.
    # If you REALLY want line breaks, you can keep <br> markers via .xpath("text()|br")
    return parts


def node_text(node) -> str:
    return norm_space(" ".join(node_text_lines(node)))


def label_key(s: str) -> str:
    s = norm_space(s)
    s = s.rstrip(":").strip()
    return s


def parse_money_to_float(s: str) -> Optional[float]:
    s = s.replace("\u00a0", " ")
    s = s.replace("€", "").replace("&euro;", "")
    s = norm_space(s)
    m = _MONEY.search(s)
    if not m:
        return None
    int_part = m.group(1).replace(" ", "")
    dec_part = m.group(2) or "0"
    try:
        return float(f"{int_part}.{dec_part}")
    except ValueError:
        return None


_MONTHS = {
    # Slovak
    "januar": 1, "január": 1,
    "februar": 2, "február": 2,
    "marec": 3,
    "april": 4, "apríl": 4,
    "maj": 5, "máj": 5,
    "jun": 6, "jún": 6,
    "jul": 7, "júl": 7,
    "august": 8,
    "september": 9,
    "oktober": 10, "október": 10,
    "november": 11,
    "december": 12,
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "october": 10, "november": 11, "december": 12,
    "sept": 9, "september": 9,
}


def parse_dodatky_cell_date(td) -> dict[str, Any]:
    """
    The date cell looks like:
      <span>31.</span><span>December</span><span>2020</span>
    """
    txts = [norm_space(t) for t in td.itertext() if norm_space(t)]
    # Often: ["31.", "December", "2020"]
    date_iso = None
    if len(txts) >= 3:
        day_s = txts[0].replace(".", "")
        month_s = txts[1].strip().lower()
        year_s = txts[2].replace(".", "")
        if month_s in _MONTHS:
            try:
                day = int(day_s)
                month = _MONTHS[month_s]
                year = int(year_s)
                date_iso = f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                date_iso = None
    return {"date_text": " ".join(txts), "date_iso": date_iso}


_RX_ID_FROM_URL = re.compile(
    r"/(?P<id>\d+)-(?P<lang>[a-z]{2})(?:/|$)(?P<rest>[^?#]*)",
    re.IGNORECASE,
)


def extract_id_from_crz_url(u: str) -> Optional[str]:
    path = urlparse(u).path
    path = unquote(path).rstrip("/")

    m = _RX_ID_FROM_URL.search(path)
    if m:
        id_ = m.group("id")
        lang = m.group("lang").lower()
        rest = (m.group("rest") or "").strip("/")
        rest = rest.replace("/", "_")
        rest = re.sub(r"_+", "_", rest).strip("_")
        return f"{id_}-{lang}" + (f"_{rest}" if rest else "")

    last = path.split("/")[-1]
    if last.isdigit():
        return last

    return None


def all_links_abs(doc, base_url: str) -> list[str]:
    hrefs = []
    for a in doc.xpath("//a[@href]"):
        h = a.get("href")
        if not h:
            continue
        hrefs.append(urljoin(base_url, h))
    # dedupe preserve order
    seen = set()
    out = []
    for u in hrefs:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def parse_card_kv(doc, header_prefix: str, *, allow_empty: bool = True) -> Optional[dict[str, str]]:
    """
    Generic card KV parser (order-preserving, duplicate-safe).
    """
    cards = doc.xpath("//div[contains(@class,'card')][.//h2[contains(@class,'card-header')]]")
    for c in cards:
        h2s = c.xpath(".//h2[contains(@class,'card-header')]")
        if not h2s:
            continue
        header = norm_space(h2s[0].text_content() or "")
        if not header.lower().startswith(header_prefix.lower()):
            continue

        kv: dict[str, str] = {}
        for li in c.xpath(".//li"):
            strong = li.xpath(".//strong")
            span = li.xpath(".//span")
            if not strong or not span:
                continue

            k = label_key(strong[0].text_content() or "")
            v = node_text(span[0])

            if not k:
                continue
            if (not v) and (not allow_empty):
                continue

            # handle duplicate labels: "IČO", etc.
            if k in kv:
                n = 2
                while f"{k} ({n})" in kv:
                    n += 1
                k = f"{k} ({n})"

            kv[k] = v

        if kv:
            return kv
    return None

def parse_identifikacia_kv(doc) -> dict[str, str]:
    """
    Identifikácia is special: it often contains two 'IČO' rows.
    We assign them based on the nearest preceding party label:
      - after Objednávateľ => 'IČO objednávateľa'
      - after Dodávateľ    => 'IČO dodávateľa'
    """
    cards = doc.xpath("//div[contains(@class,'card')][.//h2[contains(@class,'card-header')]]")
    for c in cards:
        h2s = c.xpath(".//h2[contains(@class,'card-header')]")
        if not h2s:
            continue
        header = norm_space(h2s[0].text_content() or "")
        if not header.lower().startswith("identifikácia") and not header.lower().startswith("identifikacia"):
            continue

        # first pass: collect ordered (label, value)
        items: list[tuple[str, str]] = []
        for li in c.xpath(".//li"):
            strong = li.xpath(".//strong")
            span = li.xpath(".//span")
            if not strong or not span:
                continue
            k = label_key(strong[0].text_content() or "")
            v = node_text(span[0])
            if k:
                items.append((k, v))

        # helper to find nearest prior party label for position i
        def nearest_party(i: int) -> Optional[str]:
            for j in range(i - 1, -1, -1):
                k_j = strip_diacritics(items[j][0]).lower()
                if k_j.startswith("objednavatel"):
                    return "objednavatel"
                if k_j.startswith("dodavatel"):
                    return "dodavatel"
            return None

        kv: dict[str, str] = {}
        ico_unknown_idx = 0

        for i, (k, v) in enumerate(items):
            k_norm = strip_diacritics(k).lower().strip(":").strip()
            out_k = k

            # disambiguate IČO rows
            if k_norm == "ico":
                party = nearest_party(i)
                if party == "objednavatel":
                    out_k = "IČO objednávateľa"
                elif party == "dodavatel":
                    out_k = "IČO dodávateľa"
                else:
                    ico_unknown_idx += 1
                    out_k = f"IČO #{ico_unknown_idx}"

            # preserve empty values too (e.g. Stav sometimes blank)
            if out_k in kv:
                n = 2
                while f"{out_k} ({n})" in kv:
                    n += 1
                out_k = f"{out_k} ({n})"

            kv[out_k] = v

        return kv

    return {}

def parse_attachments(doc, base_url: str) -> list[dict[str, Any]]:
    """
    Parse the 'Príloha' card list items that contain /data/att/ links.
    """
    out: list[dict[str, Any]] = []
    # locate Príloha card
    cards = doc.xpath("//div[contains(@class,'card')][.//h2[contains(@class,'card-header')]]")
    for c in cards:
        h2 = c.xpath(".//h2[contains(@class,'card-header')]")
        if not h2:
            continue
        header = norm_space(h2[0].text_content() or "")
        if header.lower() != "príloha" and header.lower() != "priloha":
            continue

        for li in c.xpath(".//li[.//a[@href]]"):
            a = li.xpath(".//a[@href]")
            if not a:
                continue
            a0 = a[0]
            href = urljoin(base_url, a0.get("href"))
            if "/data/att/" not in href:
                continue

            title = norm_space(a0.text_content() or "")
            aria = a0.get("aria-label")
            small = li.xpath(".//small")
            size_text = norm_space(small[0].text_content()) if small else ""

            # file type label if present
            typ_span = li.xpath(".//span[contains(@class,'text-uppercase')]")
            file_kind = norm_space(typ_span[0].text_content()) if typ_span else None

            filename = Path(urlparse(href).path).name

            out.append({
                "href": href,
                "filename": filename,
                "title": title,
                "aria_label": aria,
                "size_text": size_text,
                "file_kind": file_kind,
            })
        break
    return out


def parse_cenove_plnenie(doc) -> dict[str, Any]:
    """
    Parse Cenové plnenie block for 'Zmluvne dohodnutá čiastka' and 'Celková čiastka'.
    """
    cards = doc.xpath("//div[contains(@class,'card')][.//h2[contains(@class,'card-header')]]")
    for c in cards:
        h2 = c.xpath(".//h2[contains(@class,'card-header')]")
        if not h2:
            continue
        header = norm_space(h2[0].text_content() or "")
        if header.lower() != "cenové plnenie" and header.lower() != "cenove plnenie":
            continue

        txt = norm_space(c.text_content() or "")
        # crude but works on your example: find first two money occurrences
        monies = re.findall(r"\d[\d\s\u00a0]*,\d+\s*€", txt)
        agreed = parse_money_to_float(monies[0]) if len(monies) >= 1 else None
        total = parse_money_to_float(monies[1]) if len(monies) >= 2 else None
        return {"raw_text": txt, "agreed_eur": agreed, "total_eur": total}
    return {}


def parse_dodatky_table(doc, base_url: str) -> list[dict[str, Any]]:
    """
    Parse the "Dodatky:" table.
    """
    # Find h2 that starts with Dodatky
    h2s = doc.xpath("//h2")
    target_h2 = None
    for h2 in h2s:
        t = norm_space(h2.text_content() or "")
        if t.lower().startswith("dodatky"):
            target_h2 = h2
            break
    if target_h2 is None:
        return []

    # Table after that h2 (usually in following sibling div)
    table = target_h2.xpath("following::table[1]")
    if not table:
        return []
    table = table[0]

    out: list[dict[str, Any]] = []
    for tr in table.xpath(".//tbody/tr"):
        tds = tr.xpath("./td")
        if len(tds) < 5:
            continue

        date_info = parse_dodatky_cell_date(tds[0])

        # second cell: link + number in <span>
        a = tds[1].xpath(".//a[@href]")
        add_url = urljoin(base_url, a[0].get("href")) if a else None
        add_title = norm_space(a[0].text_content() or "") if a else norm_space(tds[1].text_content() or "")
        num_span = tds[1].xpath(".//span")
        add_number = norm_space(num_span[0].text_content() or "") if num_span else None

        price_txt = norm_space(tds[2].text_content() or "")
        price_eur = parse_money_to_float(price_txt)

        supplier = norm_space(tds[3].text_content() or "")
        buyer = norm_space(tds[4].text_content() or "")

        add_id = extract_id_from_crz_url(add_url) if add_url else None

        out.append({
            **date_info,
            "addendum_url": add_url,
            "addendum_id": add_id,
            "addendum_title": add_title,
            "addendum_number": add_number,
            "price_text": price_txt,
            "price_eur": price_eur,
            "supplier": supplier,
            "buyer": buyer,
        })

    return out


def parse_crz_page(page_url: str, html_text: str) -> dict[str, Any]:
    doc = html.fromstring(html_text)

    h1 = doc.xpath("//main//h1 | //h1")
    title = norm_space(h1[0].text_content() or "") if h1 else ""

    main_nodes = doc.xpath("//main")
    main_text = norm_space(main_nodes[0].text_content() if main_nodes else doc.text_content())

    links = all_links_abs(doc, page_url)

    ident = parse_identifikacia_kv(doc)  # <-- changed
    datum = parse_card_kv(doc, "Dátum", allow_empty=True) or {}
    cenove = parse_cenove_plnenie(doc)
    prilohy = parse_attachments(doc, page_url)
    dodatky = parse_dodatky_table(doc, page_url)

    page_id = None
    for k in ("ID zmluvy", "ID dodatku", "ID"):
        if k in ident:
            try:
                page_id = int(re.sub(r"\D+", "", ident[k]))
            except ValueError:
                page_id = None
            break

    if page_id is None:
        body = doc.xpath("//body")
        if body:
            cls = body[0].get("class", "")
            m = re.search(r"\bart-(\d+)\b", cls)
            if m:
                page_id = int(m.group(1))

    return {
        "url": page_url,
        "title": title,
        "main_text": main_text,
        "links": links,
        "identifikacia": ident,
        "identifikacia_ascii": dict_ascii_mirror(ident),  # <-- added
        "datum": datum,
        "cenove_plnenie": cenove,
        "prilohy": prilohy,
        "dodatky": dodatky,
        "page_id": page_id,
    }


# ---------- atomic file ops ----------
def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def atomic_download(url: str, dest: Path, *, timeout=(10, 120)) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with requests.get(url, headers={"User-Agent": UA}, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

        os.replace(tmp, dest)

    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
