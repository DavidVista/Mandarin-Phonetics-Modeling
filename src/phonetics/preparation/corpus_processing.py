"""Shared deterministic corpus transformations used by legacy and staged flows."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict

from ..data_contract import FILTERS, pair_reason, sample_id
from ..pinyin import canonical_labels


class CorpusConflictError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def normalize_rows(rows, tone_convention):
    normalized=[]
    for row in rows:
        error=None
        try: labels=canonical_labels(row["raw_pinyin"],tone_convention=tone_convention) if row["raw_pinyin"] else []
        except ValueError as exc: labels=list(row["raw_pinyin"]);error=str(exc)
        normalized.append({**row,"labels":labels,"normalization_error":error})
    return normalized,{"input_count":len(rows),"output_count":len(normalized),"rejected":0}


def filter_rows(rows,wubi,acoustic,stages,stages_to_run=None):
    active,rejected_rows=list(rows),[]
    for stage in FILTERS if stages_to_run is None else stages_to_run:
        started=time.perf_counter();kept=[]
        for row in active:
            failed=stage=="pinyin" and row.get("normalization_error")
            if failed or pair_reason(row["text"],row["labels"],wubi,acoustic,stage=stage):
                detail={"reason":stage,**row}
                if stage=="wubi":detail["missing_characters"]=sorted(set(row["text"])-set(wubi))
                if stage=="acoustic":detail["missing_tokens"]=sorted(set(row["labels"])-set(acoustic))
                rejected_rows.append(detail)
            else:kept.append(row)
        stages.append({"name":stage,"unit":"rows","input_count":len(active),"output_count":len(kept),
                       "rejected":len(active)-len(kept),"status":"success","counts_complete":True,
                       "duration_seconds":time.perf_counter()-started})
        active=kept
    return active,rejected_rows


def _conflict_groups(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row["text"]].append(row)
    conflicts=[]
    for text,members in sorted(grouped.items()):
        variants=defaultdict(list)
        for row in members:variants[tuple(row["labels"])].append(row)
        if len(variants)>1:
            conflicts.append({"text":text,"members":members,"label_variants":[{"labels":list(labels),
                "source_ids":sorted({r["source_id"] for r in variant}),
                "source_rows":sorted(r["source_row"] for r in variant)} for labels,variant in sorted(variants.items())]})
    return grouped,conflicts


def exclude_conflicts(rows):
    grouped,conflicts=_conflict_groups(rows);conflicting_texts={group["text"] for group in conflicts};rejections=[]
    for group in conflicts:
        variants=group["label_variants"]
        for row in group["members"]:
            rejections.append({"reason":"conflicting_pronunciation",**row,"label_variants":variants})
    kept=[row for row in rows if row["text"] not in conflicting_texts]
    return kept,rejections,{"input_count":len(rows),"output_count":len(kept),"rejected":len(rejections),
                            "conflict_groups":len(conflicts),"conflict_rows":len(rejections)}


def deduplicate_rows(rows, *, conflict_policy="error"):
    if conflict_policy not in {"error","exclude"}:raise ValueError("unsupported conflict policy")
    grouped,conflicts=_conflict_groups(rows)
    if conflicts:
        if conflict_policy=="exclude":
            kept,_rejections,_counts=exclude_conflicts(rows)
            return deduplicate_rows(kept,conflict_policy="error")
        conflict_rows=sum(sum(len(v["source_rows"]) for v in g["label_variants"]) for g in conflicts)
        raise CorpusConflictError("conflicting labels for duplicate text",{"input_count":len(rows),"output_count":0,
            "rejected":conflict_rows,"blocked_count":len(rows)-conflict_rows,"conflict_groups":len(conflicts),
            "conflict_rows":conflict_rows,"conflicts":[{k:v for k,v in group.items() if k!="members"} for group in conflicts],"counts_complete":True})
    result=[]
    for members in grouped.values():
        row={**members[0],"source_ids":sorted({m["source_id"] for m in members})}
        row["sample_id"]=sample_id(row["text"],row["labels"]);result.append(row)
    result.sort(key=lambda row:row["sample_id"])
    return result,{"input_count":len(rows),"output_count":len(result),"rejected":len(rows)-len(result),
                   "duplicates":len(rows)-len(result),"conflict_groups":0,"conflict_rows":0}


def sample_rows(rows,spec):
    size=spec["size"]
    if size is not None and size>len(rows):raise ValueError("sampling size exceeds eligible pool")
    selected=list(rows) if size is None else random.Random(spec["seed"]).sample(rows,size)
    selected.sort(key=lambda row:row["sample_id"])
    return selected,{"input_count":len(rows),"output_count":len(selected),"rejected":len(rows)-len(selected)}


def coverage(rows):
    chars=Counter(ch for row in rows for ch in row["text"]);labels=Counter(x for row in rows for x in row["labels"]);readings=defaultdict(set)
    for row in rows:
        for character,label in zip(row["text"],row["labels"]):readings[character].add(label)
    return {"rows":len(rows),"characters":sum(chars.values()),"lengths":dict(sorted(Counter(len(r["text"]) for r in rows).items())),
            "unique_characters":len(chars),"pinyin":dict(sorted(labels.items())),
            "tones":dict(sorted(Counter(x[-1] for x in labels.elements()).items())),
            "neutral":sum(v for k,v in labels.items() if k.endswith("5")),
            "polyphonic_readings":{k:sorted(v) for k,v in sorted(readings.items()) if len(v)>1}}
