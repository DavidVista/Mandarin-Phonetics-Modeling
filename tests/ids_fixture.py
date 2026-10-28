"""Small pinned dictionaries for offline preparation tests."""
import json
from pathlib import Path
import shutil

from phonetics.artifacts import sha256_file


def ids_pin(root, characters):
    root = Path(root)
    path = root / "dictionary.txt"
    path.write_text("".join(json.dumps({"character": char, "decomposition": "？"},
                                     ensure_ascii=False) + "\n"
                            for char in sorted(set(characters), key=ord)), encoding="utf-8")
    for name in ("COPYING", "LGPL"):
        shutil.copyfile(Path(__file__).parents[1] / "datasets/ids-decomposition" / name,
                        root / name)
    return {"path": str(path), "sha256": sha256_file(path),
            "snapshot_commit": "618dbab8a8ddefb958763c8b4afbaa741a4460de",
            "license": "LGPL-3.0-or-later", "attribution": "Synthetic IDS test fixture"}
