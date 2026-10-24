"""Export a pinned DuYu split without generating or guessing pinyin labels."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from ..artifacts import sha256_file,write_json,write_jsonl
from ..data_contract import is_han
from ..pinyin import canonical_labels,decompose


def export_duyu(output_dir,*,revision,split,columns,tone_convention,dataset_loader=None,allow_download=False):
    _validate_args(revision,split,columns,tone_convention)
    target=Path(output_dir).resolve()
    if target.exists(): raise FileExistsError(target)
    if dataset_loader is None:
        if not allow_download: raise PermissionError("real DuYu loading requires allow_download=True")
        try: from datasets import load_dataset
        except ImportError as exc: raise RuntimeError("datasets is required for real DuYu loading") from exc
        rows=load_dataset("Duyu/Pinyin-Hanzi",revision=revision,split=split)
        loader_kind="datasets"
    else:
        rows=dataset_loader("Duyu/Pinyin-Hanzi",revision=revision,split=split)
        loader_kind="injected"
    parent=target.parent;parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=parent))
    accepted=[];rejected=[];raw_rows=[]
    try:
        for index,row in enumerate(rows):
            if not isinstance(row,dict): raise ValueError(f"DuYu row {index} is not an object")
            required={columns["text"],columns["pinyin"]}
            if columns.get("id"): required.add(columns["id"])
            missing=required-set(row)
            if missing: raise ValueError(f"DuYu row {index} is missing mapped columns: {sorted(missing)}")
            raw_text=row.get(columns["text"]); raw_pinyin=row.get(columns["pinyin"]); original_id=row.get(columns.get("id")) if columns.get("id") else None
            source_id=str(original_id) if original_id not in (None,"") else f"{revision}:{split}:{index}"
            base={"source_id":source_id,"source_row":index,"raw_text":raw_text,"raw_pinyin":raw_pinyin}
            cleaned,labels,reason=_normalize(raw_text,raw_pinyin,tone_convention)
            raw_rows.append({**base,"text":cleaned,"labels":labels,"status":"accepted" if reason is None else "rejected","reason":reason})
            if reason: rejected.append({**base,"reason":reason})
            else: accepted.append({"text":cleaned,"pinyin":" ".join(labels),"source_id":source_id})
        with (temp/"corpus.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["text","pinyin","source_id"]);writer.writeheader();writer.writerows(accepted)
        write_jsonl(temp/"source_rows.jsonl",raw_rows);write_jsonl(temp/"rejections.jsonl",rejected)
        counts={"input":len(raw_rows),"accepted":len(accepted),"rejected":len(rejected),"reasons":dict(sorted(Counter(r["reason"] for r in rejected).items()))}
        if counts["input"]!=counts["accepted"]+counts["rejected"]: raise AssertionError("export count imbalance")
        write_json(temp/"metrics.json",counts)
        files={p.name:sha256_file(p) for p in temp.iterdir() if p.is_file()}
        write_json(temp/"provenance.json",{"schema_version":1,"dataset":"Duyu/Pinyin-Hanzi","revision":revision,"split":split,
            "columns":columns,"tone_convention":tone_convention,"loader":loader_kind,"files":files})
        os.rename(temp,target);return target
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise


def _validate_args(revision,split,columns,tone_convention):
    if (not isinstance(revision,str) or not revision.strip() or "REPLACE_WITH" in revision
            or revision.strip().lower() in {"latest","main","master"}): raise ValueError("exact non-placeholder revision required")
    if not isinstance(split,str) or not split.strip() or "REPLACE_WITH" in split: raise ValueError("exact non-placeholder split required")
    if not isinstance(columns,dict) or not {"text","pinyin"}.issubset(columns) or set(columns)-{"text","pinyin","id"}: raise ValueError("explicit text/pinyin columns required")
    if any(not isinstance(value,str) or not value or "REPLACE_WITH" in value for value in columns.values()): raise ValueError("column mapping contains a placeholder")
    if len(set(columns.values()))!=len(columns): raise ValueError("column mappings must be distinct")
    if tone_convention not in {"explicit","neutral_unmarked"}: raise ValueError("unsupported tone convention")


def _normalize(text,pinyin,tone_convention):
    if not isinstance(text,str) or not text: return None,None,"empty"
    if any(ch.isdigit() for ch in text): return None,None,"digit"
    raw=pinyin.split() if isinstance(pinyin,str) else list(pinyin) if isinstance(pinyin,(list,tuple)) else None
    if raw is None or len(text)!=len(raw): return None,None,"alignment"
    kept=[(char,label) for char,label in zip(text,raw) if not char.isspace() and unicodedata.category(char)[0]!="P"]
    cleaned="".join(char for char,_ in kept); labels=[label for _,label in kept]
    if not cleaned: return None,None,"empty_after_cleaning"
    if any(not is_han(char) for char in cleaned): return None,None,"unsupported_character"
    try:
        labels=canonical_labels(labels,tone_convention=tone_convention)
        for label in labels: decompose(label)
    except (ValueError,TypeError,IndexError): return None,None,"pinyin"
    return cleaned,labels,None
