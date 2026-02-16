#!/usr/bin/env python3
from __future__ import annotations

import calendar
import datetime as dt
import os
import re
import time
import urllib.request
import zipfile
import itertools
import shutil
import http.client
import ssl
import urllib.error

from crz_config import BASE_START, DUMP_DIR

YESTERDAY = (dt.datetime.now() - dt.timedelta(days=1)).date()

START_DATE_FILE = ".start_date.txt"

DATE_RX = re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})(?:\..*)?$")


def _extract_dmy(text: str) -> tuple[int, int, int] | None:
    nums = [int(s) for s in re.findall(r"\b\d+\b", text)]
    if len(nums) != 3:
        return None
    day, month, year = nums[0], nums[1], nums[2]
    return day, month, year


def clamp_date_dmy(
    day: int,
    month: int,
    year: int,
    *,
    min_date: dt.date,
    max_date: dt.date,
) -> dt.date:
    year = max(min_date.year, min(max_date.year, year))
    month = max(1, min(12, month))
    last_day = calendar.monthrange(year, month)[1]
    day = max(1, min(last_day, day))
    d = dt.date(year, month, day)
    if d < min_date:
        return min_date
    if d > max_date:
        return max_date
    return d


def prompt_date(
    prompt: str,
    *,
    default: dt.date,
    min_date: dt.date,
    max_date: dt.date,
    allow_beginning: bool = False,
) -> dt.date:
    s = input(prompt).strip()
    if not s:
        return default

    if s.lower() == "y":
        return max_date

    if allow_beginning and s.lower() == "b":
        return min_date

    dmy = _extract_dmy(s)
    if dmy is None:
        print(f"Invalid input, using default date: {default}")
        return default

    day, month, year = dmy
    return clamp_date_dmy(day, month, year, min_date=min_date, max_date=max_date)


def date_range(start: dt.date, end: dt.date) -> list[str]:
    n = (end - start).days
    return [(start + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n + 1)]


def download_with_retries(
    url: str,
    dest_path: str,
    *,
    timeout_s: float = 60.0,
    backoff_start_s: float = 2.0,
    backoff_max_s: float = 300.0,
    retry_forever: bool = True,
) -> bool:
    """
    Returns True if downloaded, False if skipped (e.g., 404).
    Retries transient failures with exponential backoff.
    """
    tmp_path = dest_path + ".part"

    for attempt in itertools.count(1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CRZ-scraper/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(resp, f)

            os.replace(tmp_path, dest_path)
            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return False

            if not retry_forever:
                raise

            wait = min(backoff_max_s, backoff_start_s * (2 ** min(attempt - 1, 10)))
            print(f" HTTP {e.code}; retrying in {wait:.0f}s...", end="")
            time.sleep(wait)

        except urllib.error.URLError as e:
            if not retry_forever:
                raise

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            wait = min(backoff_max_s, backoff_start_s * (2 ** min(attempt - 1, 10)))
            print(f" URLError({e.reason!r}); retrying in {wait:.0f}s...", end="")
            time.sleep(wait)

        except (
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            ConnectionResetError,
            BrokenPipeError,
            ssl.SSLError,
        ) as e:
            if not retry_forever:
                raise

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            wait = min(backoff_max_s, backoff_start_s * (2 ** min(attempt - 1, 10)))
            print(f" {type(e).__name__}({e}); retrying in {wait:.0f}s...", end="")
            time.sleep(wait)


def read_saved_start_date(dump_dir: str) -> dt.date | None:
    p = os.path.join(dump_dir, START_DATE_FILE)
    if not os.path.isfile(p):
        return None
    try:
        s = open(p, "r", encoding="utf-8").read().strip()
        d = dt.date.fromisoformat(s)
        return d
    except Exception:
        return None


def write_saved_start_date(dump_dir: str, start_date: dt.date) -> None:
    p = os.path.join(dump_dir, START_DATE_FILE)
    with open(p, "w", encoding="utf-8") as f:
        f.write(start_date.isoformat() + "\n")


def existing_dates_in_dump(dump_dir: str) -> set[str]:
    """
    Returns set of date strings YYYY-MM-DD found as filename prefixes inside dump_dir.
    We count any file named like YYYY-MM-DD.xml (or YYYY-MM-DD.anything).
    """
    out: set[str] = set()
    if not os.path.isdir(dump_dir):
        return out

    for name in os.listdir(dump_dir):
        if name == START_DATE_FILE:
            continue
        m = DATE_RX.match(name)
        if m:
            out.add(m.group("d"))
    return out


def extract_zip(zip_path: str, dump_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dump_dir)
    os.remove(zip_path)


def main() -> None:
    print("*** Downloading DB of contracts from crz.gov.sk. ***\n")

    first_run = not os.path.isdir(DUMP_DIR)
    os.makedirs(DUMP_DIR, exist_ok=True)

    # Decide start date:
    saved = read_saved_start_date(DUMP_DIR)

    if first_run or not os.listdir(DUMP_DIR) or saved is None:
        # First run (or no saved start date): ask once, then persist.
        start_date = prompt_date(
            "Start date ('d.m.yyyy', 'b' for beginning, 'y' for yesterday): ",
            default=BASE_START,
            min_date=BASE_START,
            max_date=YESTERDAY,
            allow_beginning=True,
        )
        write_saved_start_date(DUMP_DIR, start_date)
        print(f"Using start date: {start_date} (saved to {DUMP_DIR}/{START_DATE_FILE})")
    else:
        start_date = saved
        print(f"Using saved start date: {start_date} (from {DUMP_DIR}/{START_DATE_FILE})")

    end_date = YESTERDAY
    print(f"Downloading up to: {end_date}\n")

    have = existing_dates_in_dump(DUMP_DIR)

    downloaded = 0
    skipped_existing = 0
    skipped_404 = 0
    extracted_existing_zip = 0

    for date_str in date_range(start_date, end_date):
        # Already have extracted file(s) for this date -> skip
        if date_str in have:
            print(f"Downloading date : {date_str} ...SKIP (already present)")
            skipped_existing += 1
            continue

        zip_path = os.path.join(DUMP_DIR, f"{date_str}.zip")

        # If a zip is already here (interrupted previous run), just extract it
        if os.path.isfile(zip_path):
            print(f"Downloading date : {date_str} ...FOUND ZIP (extracting)")
            try:
                extract_zip(zip_path, DUMP_DIR)
                have.add(date_str)
                extracted_existing_zip += 1
            except Exception as e:
                print(f" ...FAILED extracting existing zip: {e!r}")
            continue

        print(f"Downloading date : {date_str}", end="")

        url = f"http://www.crz.gov.sk/export/{date_str}.zip"
        ok = download_with_retries(url, zip_path, timeout_s=60.0, retry_forever=True)

        if not ok:
            print(" ...404 (not found), skipping...")
            skipped_404 += 1
            continue

        print(" ...OK (downloaded)")
        try:
            extract_zip(zip_path, DUMP_DIR)
            have.add(date_str)
            downloaded += 1
        except Exception as e:
            print(f" ...FAILED extracting: {e!r}")

    print("\nDone.")
    print(f"Downloaded new days: {downloaded}")
    print(f"Extracted previously-downloaded zips: {extracted_existing_zip}")
    print(f"Skipped (already present): {skipped_existing}")
    print(f"Skipped (404): {skipped_404}")


if __name__ == "__main__":
    main()
