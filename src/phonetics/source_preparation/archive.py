"""Safe flat sound archive extraction with checksum validation."""

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
