from pathlib import Path
import datetime as dt

BASE_START = dt.date(2011, 1, 1)

DIR_ROOT = Path("data")

DUMP_DIR                = DIR_ROOT / Path("xml_dump")
OUT_JSON_DIR            = DIR_ROOT / Path("json_dump")
BAD_XML_DIR             = DIR_ROOT / Path("corrupted_XML_files")
ATT_DIR                 = DIR_ROOT / Path("attachments")
PARSED_CONTRACTS_DIR    = DIR_ROOT / Path("contracts_parsed")
PARSED_AMENDMENTS_DIR   = DIR_ROOT / Path("amendments_parsed")
PAGES_DIR               = DIR_ROOT / Path("fetched_pages")
RESOLVED_COLLISIONS_DIR = DIR_ROOT / Path("resolved_collisions")

FILTERED_CONTRACTS_DIR = DIR_ROOT / Path("filtered_contracts")

CRZ_BASE_URL = "https://www.crz.gov.sk/"
