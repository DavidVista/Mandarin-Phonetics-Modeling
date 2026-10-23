"""Strict, provenance-preserving AIxBlock lot ingestion."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

from ..artifacts import sha256_file
from ..data_contract import is_han

EXPECTED = {"Utterance", "Status", "Language"}


def clean_aixblock_text(value):
    if not isinstance(value, str) or not value:
        return None, "missing_or_empty", False
    if any(ch.isdigit() for ch in value):
        return None, "contains_digit", False
    cleaned = "".join(ch for ch in value if not ch.isspace() and unicodedata.category(ch)[0] != "P")
    if not cleaned:
        return None, "empty_after_cleaning", cleaned != value
    if any(not is_han(ch) for ch in cleaned):
        return None, "unsupported_character", cleaned != value
    return cleaned, None, cleaned != value


def load_aixblock_lots(lots):
    if not isinstance(lots, (list, tuple)) or len(lots) != 2 or {x.get("lot_id") for x in lots} != {"lot1", "lot2"}:
        raise ValueError("exactly lot1 and lot2 are required")
    records, rejections, changed, loaded = [], [], 0, 0
    for spec in lots:
        if set(spec) != {"lot_id", "path", "sha256", "revision"} or not spec["revision"]:
            raise ValueError("lot requires lot_id,path,sha256,revision")
        path=Path(spec["path"]).resolve(); digest=sha256_file(path)
        if digest != spec["sha256"]: raise ValueError(f"lot checksum mismatch: {spec['lot_id']}")
        rows=_read_rows(path)
        seen_ids=set()
        for row_number,row in enumerate(rows,2):
            loaded += 1
            raw=row.get("Utterance"); cleaned,reason,was_changed=clean_aixblock_text(raw)
            original_id=row.get("Sr No")
            if original_id in (None,""): original_id=row.get("SrNo")
            stable_part=original_id if original_id not in (None,"") else f"row-{row_number}"
            if stable_part in seen_ids: stable_part=f"{stable_part}:row-{row_number}"
            seen_ids.add(stable_part); source_row=row_number
            source_id=f"{spec['lot_id']}:{digest[:16]}:{stable_part}"
            base={"source_id":source_id,"source_row":source_row,"lot_id":spec["lot_id"],
                  "source_original_id":original_id,"source_sha256":digest,"source_revision":spec["revision"],"raw_text":raw}
            if row.get("Status") != "Accepted": reason="status_not_accepted"
            elif row.get("Language") != "Simplified Chinese Mainland": reason="language_not_target"
            if reason: rejections.append({**base,"reason":reason})
            else:
                changed += int(was_changed); records.append({**base,"text":cleaned})
    return records,rejections,{"input":loaded,"source_eligible":sum(x["reason"] not in {"status_not_accepted","language_not_target"} for x in rejections)+len(records),
        "accepted":len(records),"rejected":len(rejections),"changed":changed}


def _read_rows(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle,strict=True)
            if not reader.fieldnames or not EXPECTED.issubset(reader.fieldnames) or len(reader.fieldnames)!=len(set(reader.fieldnames)):
                raise ValueError("AIxBlock columns must include Utterance,Status,Language")
            rows=list(reader)
            if any(None in row for row in rows): raise ValueError("malformed AIxBlock CSV row")
            return rows
    if path.suffix.lower() in {".xlsx",".xls"}:
        try: import pandas as pd
        except ImportError as exc: raise RuntimeError("pandas/openpyxl are required for XLSX") from exc
        frame=pd.read_excel(path)
        if not EXPECTED.issubset(frame.columns): raise ValueError("AIxBlock columns missing")
        return frame.where(frame.notna(),None).to_dict("records")
    raise ValueError("lot must be CSV or XLSX")
