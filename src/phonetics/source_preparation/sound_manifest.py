"""Build an audited canonical-label manifest for a local sound directory."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

from ..artifacts import sha256_file,write_json
from ..pinyin import canonical_labels,decompose
from .archive import ALLOWED,_validate_names

CHINESE_LESSONS_ALIASES={
    "lue4.mp3":"lve4","nue4.mp3":"nve4","luu2.mp3":"lv2","luu3.mp3":"lv3",
    "luu4.mp3":"lv4","nuu3.mp3":"nv3","nuu4.mp3":"nv4",
}


def build_sound_manifest(sound_dir,output_dir,*,aliases=None,tone_convention="neutral_unmarked"):
    source=Path(sound_dir).resolve();target=Path(output_dir).resolve()
    if target.exists(): raise FileExistsError(target)
    mapping=dict(CHINESE_LESSONS_ALIASES if aliases is None else aliases)
    if any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in mapping.items()):
        raise ValueError("aliases must map non-empty filenames to non-empty labels")
    files=sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED),key=lambda p:p.name)
    if not files: raise ValueError("sound source is empty")
    _validate_names([p.name for p in files])
    unused=set(mapping)-{p.name for p in files}
    if unused: raise ValueError(f"unused sound aliases: {sorted(unused)}")
    included=[];excluded=[];labels={}
    for path in files:
        candidate=mapping.get(path.name,path.stem)
        try:
            label=canonical_labels([candidate],tone_convention=tone_convention)[0]
            decompose(label)
        except (TypeError,ValueError,IndexError) as exc:
            excluded.append({"path":path.name,"raw_label":path.stem,"candidate":candidate,
                             "reason":"unsupported_pinyin","detail":str(exc)})
            continue
        if label in labels:
            raise ValueError(f"canonical label collision {label}: {labels[label]} and {path.name}")
        labels[label]=path.name
        included.append({"path":path.name,"pinyin":label,"raw_label":path.stem,
                         "mapped":path.name in mapping,"sha256":sha256_file(path),"size":path.stat().st_size})
    parent=target.parent;parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=parent))
    try:
        manifest=temp/"sounds_manifest.csv"
        with manifest.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["path","pinyin"],lineterminator="\n");writer.writeheader()
            writer.writerows(({"path":r["path"],"pinyin":r["pinyin"]} for r in included))
        report={"schema_version":1,"input":len(files),"included":len(included),"excluded":len(excluded),
                "counts_balanced":len(files)==len(included)+len(excluded),"excluded_records":excluded}
        write_json(temp/"sound_manifest_report.json",report)
        provenance={"schema_version":1,"source_dir":str(source),"tone_convention":tone_convention,
                    "aliases":dict(sorted(mapping.items())),"files":included,
                    "manifest_sha256":sha256_file(manifest)}
        write_json(temp/"sounds_provenance.json",provenance)
        os.rename(temp,target)
        return target
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise
