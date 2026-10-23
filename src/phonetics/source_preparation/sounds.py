"""Stage a fixed StudyCLI inventory plus explicitly selected historical neutral WAVs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import wave
from pathlib import Path

from ..artifacts import sha256_file, write_json
from ..pinyin import canonical_labels, decompose


def prepare_studycli_sounds(*, source_root, tonal_inventory_path, inventory_sha256,
                            neutral_dir, requested_urls, allow_network=False, fetcher=None,
                            decoder=None, require_complete=True):
    inventory_path=Path(tonal_inventory_path).resolve()
    if sha256_file(inventory_path)!=inventory_sha256: raise ValueError("inventory checksum mismatch")
    tokens=[line.strip() for line in inventory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tokens)!=len(set(tokens)): raise ValueError("duplicate fixed inventory token")
    for token in tokens:
        if token[-1:] not in "12345" or canonical_labels([token],tone_convention="explicit")[0]!=token: raise ValueError("inventory requires canonical explicit tones")
        decompose(token)
    neutral_dir=Path(neutral_dir).resolve(); neutral={p.stem:p for p in sorted(neutral_dir.glob("*.wav"))}
    if any(key[-1:]!="5" or canonical_labels([key],tone_convention="explicit")[0]!=key for key in neutral):
        raise ValueError("historical neutral directory contains a non-canonical tone-5 WAV")
    if set(neutral)-set(tokens): raise ValueError("fixed inventory does not cover existing neutral keys")
    if set(requested_urls) & set(neutral): raise ValueError("requested sounds collide with historical neutral sounds")
    unknown=set(requested_urls)-set(tokens)
    if unknown: raise ValueError(f"requested keys outside fixed inventory: {sorted(unknown)}")
    source_root=Path(source_root).resolve(); source_root.mkdir(parents=True,exist_ok=True); root=source_root/"audio"
    fetch_mode="injected" if fetcher is not None else "network"
    identity={"inventory_sha256":inventory_sha256,"fixed_tokens":tokens,"requested_urls":dict(sorted(requested_urls.items())),"fetch_mode":fetch_mode,
              "neutral_sha256":{key:sha256_file(path) for key,path in neutral.items()},"require_complete":require_complete}
    identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if root.exists(): return _validate_existing(root,identity_sha256)
    parent=root.parent; temp=Path(tempfile.mkdtemp(prefix=".audio.",dir=parent))
    (temp/"sounds").mkdir(); provenance=[]; attempts=[]
    try:
        for token,path in neutral.items():
            info=_decode(path,decoder); destination=temp/"sounds"/path.name; shutil.copyfile(path,destination)
            if sha256_file(destination)!=sha256_file(path): raise ValueError("neutral copy checksum mismatch")
            provenance.append(_entry(token,destination,"legacy_synthetic_neutral",info,
                {"attribution":"historical mandarin-sounds-dataset notebook; speaker acoustics not independently verified"}))
        for token,url in sorted(requested_urls.items()):
            destination=temp/"sounds"/f"{token}{Path(url).suffix.lower() or '.mp3'}"
            temporary=None
            try:
                body=_fetch(url,allow_network,fetcher)
                temporary=destination.with_suffix(destination.suffix+".tmp"); temporary.write_bytes(body)
                info=_decode(temporary,decoder); os.rename(temporary,destination)
                provenance.append(_entry(token,destination,"studycli",info,{"url":url}))
                attempts.append({"pinyin":token,"url":url,"status":"success","bytes":len(body)})
            except Exception as exc:
                if temporary is not None and temporary.exists(): temporary.unlink()
                attempts.append({"pinyin":token,"url":url,"status":"failure","error":type(exc).__name__,"message":str(exc)})
        actual={row["pinyin"] for row in provenance}; missing=sorted(set(tokens)-actual)
        if missing and require_complete: raise ValueError(f"fixed inventory incomplete: {missing}")
        with (temp/"sounds_manifest.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["path","pinyin"]); writer.writeheader()
            writer.writerows({"path":Path(row["path"]).name,"pinyin":row["pinyin"]} for row in sorted(provenance,key=lambda x:x["pinyin"]))
        write_json(temp/"sounds_provenance.json",{"schema_version":1,"identity_sha256":identity_sha256,**identity,
            "manifest_sha256":sha256_file(temp/"sounds_manifest.csv"),"files":provenance,"attempts":attempts,
            "counts":{"requested":len(tokens),"valid":len(actual),"missing":len(missing),"failed_attempts":sum(x["status"]=="failure" for x in attempts)},
            "complete":not missing,"missing":missing})
        os.rename(temp,root); return root
    except Exception as exc:
        write_json(source_root/"audio.failed.json",{"schema_version":1,"identity_sha256":identity_sha256,"status":"failure",
            "exception":type(exc).__name__,"message":str(exc),"attempts":attempts,
            "counts":{"attempted":len(attempts),"valid":len(provenance),"failed":sum(x["status"]=="failure" for x in attempts)}})
        shutil.rmtree(temp,ignore_errors=True); raise


def _fetch(url,allow_network,fetcher):
    if fetcher is None:
        if not allow_network: raise PermissionError("network access requires allow_network=True")
        with urllib.request.urlopen(url,timeout=20) as response:
            if response.status!=200: raise ValueError(f"HTTP {response.status}")
            return response.read()
    result=fetcher(url)
    if isinstance(result,tuple):
        status,body=result
        if status!=200: raise ValueError(f"HTTP {status}")
        return body
    return result


def _decode(path,decoder):
    if decoder is not None: return decoder(path)
    if path.name.lower().endswith((".wav", ".wav.tmp")):
        with wave.open(str(path),"rb") as handle:
            rate,channels,frames,width=handle.getframerate(),handle.getnchannels(),handle.getnframes(),handle.getsampwidth(); data=handle.readframes(frames)
        if min(rate,channels,frames,width)<=0 or len(data)!=frames*channels*width: raise ValueError("invalid or truncated WAV")
        return {"sample_rate":rate,"channels":channels,"frames":frames}
    try: import soundfile as sf
    except ImportError as exc: raise RuntimeError("soundfile is required to decode non-WAV audio") from exc
    data,rate=sf.read(str(path),always_2d=True,dtype="float32")
    import numpy as np
    if data.size<=0 or rate<=0 or not np.isfinite(data).all(): raise ValueError("invalid audio")
    info=sf.info(str(path))
    if len(data)!=info.frames: raise ValueError("truncated audio decode")
    return {"sample_rate":info.samplerate,"channels":info.channels,"frames":info.frames}


def _entry(token,path,kind,info,extra):
    return {"pinyin":token,"path":path.name,"sha256":sha256_file(path),"size":path.stat().st_size,
            "source_kind":kind,**info,**extra}


def _validate_existing(root,identity_sha256):
    provenance=json.loads((root/"sounds_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("identity_sha256")!=identity_sha256: raise ValueError("existing sound source configuration mismatch")
    manifest=root/"sounds_manifest.csv"
    if not manifest.is_file() or sha256_file(manifest)!=provenance.get("manifest_sha256"): raise ValueError("existing sound manifest changed")
    with manifest.open(encoding="utf-8",newline="") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=["path","pinyin"]: raise ValueError("existing sound manifest schema changed")
        rows=list(reader)
    expected={(row["path"],row["pinyin"]) for row in provenance.get("files",[])}
    actual={(row.get("path"),row.get("pinyin")) for row in rows}
    if len(rows)!=len(actual) or actual!=expected: raise ValueError("existing sound manifest semantics changed")
    names={row[0] for row in actual}
    if any(Path(name).name!=name for name in names) or {p.name for p in (root/"sounds").iterdir() if p.is_file()}!=names:
        raise ValueError("existing sound file set changed")
    fixed=set(provenance.get("fixed_tokens",[])); valid={row[1] for row in actual}; missing=sorted(fixed-valid)
    if provenance.get("missing")!=missing or provenance.get("complete")!=(not missing): raise ValueError("existing completeness report changed")
    for row in provenance.get("files",[]):
        path=root/"sounds"/Path(row["path"]).name
        if not path.is_file() or sha256_file(path)!=row["sha256"]: raise ValueError("existing sound file changed")
    return root
