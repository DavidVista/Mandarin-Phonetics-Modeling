"""Deterministic flat ZIP packing and fail-before-write extraction."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path,PurePosixPath,PureWindowsPath

from ..artifacts import sha256_file

ALLOWED={".wav",".mp3"}
WINDOWS_FORBIDDEN=set('<>:"/\\|?*')
WINDOWS_DEVICES={"con","prn","aux","nul",*(f"com{i}" for i in range(1,10)),*(f"lpt{i}" for i in range(1,10))}


def pack_sounds(source_dir,archive_path):
    source=Path(source_dir).resolve();target=Path(archive_path).resolve()
    if target.exists(): raise FileExistsError(target)
    files=sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED),key=lambda p:p.name)
    if not files: raise ValueError("sound source is empty")
    _validate_names([p.name for p in files])
    target.parent.mkdir(parents=True,exist_ok=True);fd,temp_name=tempfile.mkstemp(prefix=f".{target.name}.",dir=target.parent);os.close(fd);temp=Path(temp_name)
    try:
        with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=(0o100644<<16)
                archive.writestr(info,path.read_bytes())
        os.rename(temp,target)
        return {"path":str(target),"sha256":sha256_file(target),"files":len(files),"bytes":sum(p.stat().st_size for p in files),
                "inventory":{p.name:sha256_file(p) for p in files}}
    except Exception:
        temp.unlink(missing_ok=True);raise


def unpack_sounds(archive_path,destination,*,expected_sha256):
    archive_path=Path(archive_path).resolve();destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    actual_sha=sha256_file(archive_path)
    if actual_sha!=expected_sha256: raise ValueError("archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos=archive.infolist();_validate_entries(infos)
        inventory={info.filename:hashlib.sha256(archive.read(info)).hexdigest() for info in infos}
        payload={info.filename:archive.read(info) for info in infos}
    parent=destination.parent;parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix=f".{destination.name}.",dir=parent))
    try:
        for name,data in payload.items():
            path=temp/name;path.write_bytes(data)
            if sha256_file(path)!=inventory[name]: raise ValueError("extracted checksum mismatch")
        os.rename(temp,destination)
        return {"path":str(destination),"archive_sha256":actual_sha,"files":len(payload),"bytes":sum(len(x) for x in payload.values()),"inventory":inventory}
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise


def _validate_entries(infos):
    if not infos: raise ValueError("sound archive is empty")
    names=[]
    for info in infos:
        name=info.filename
        mode=(info.external_attr>>16)&0xFFFF
        if stat.S_ISLNK(mode): raise ValueError(f"symlink ZIP entry: {name}")
        names.append(name)
    _validate_names(names)


def _validate_names(names):
    for name in names:
        posix=PurePosixPath(name);windows=PureWindowsPath(name)
        device=name.split(".",1)[0].casefold()
        if (not name or name!=posix.name or "\\" in name or posix.is_absolute() or windows.is_absolute()
                or windows.drive or len(posix.parts)!=1 or any(c in WINDOWS_FORBIDDEN for c in name)
                or name.endswith(("."," ")) or device in WINDOWS_DEVICES or Path(name).suffix.lower() not in ALLOWED):
            raise ValueError(f"unsafe or unsupported sound filename: {name}")
    if len(names)!=len(set(names)): raise ValueError("duplicate ZIP entries")
    if len(names)!=len({name.casefold() for name in names}): raise ValueError("case-colliding ZIP entries")
