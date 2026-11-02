"""Create a separate presentation run from completed numerical evaluation artifacts."""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from ..evaluation.evaluation_outputs import (
    _evaluation_run, _write, load_evaluation_outputs, load_random_control_summary,
)
from ..evaluation.evaluation_tables import render_evaluation_tables
from ..research.artifacts import ensure_output_outside_inputs, file_sha256, safe_run_id
from . import apply_style, load_protocol
from .evaluation import FIGURE_STEMS, _check_paired_outputs, _guard_not_in_immutable_run, render_evaluation


def write_evaluation_report(project_root, run_id, *, anchor_q_output, dimsim_output,
                            random_control_output, source_hashes, source_digest,
                            resolved_config=None, protocol=None):
    """Save tables and PNG/EPS figures without checkpoints, extraction or scoring."""
    run_id = safe_run_id(run_id)
    if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("source_digest must be an explicit lowercase SHA-256 digest")
    inputs = [Path(path).resolve(strict=True) for path in
              (anchor_q_output, dimsim_output, random_control_output)]
    target = Path(project_root).resolve() / "outputs" / "contextual-evaluation-report" / run_id
    ensure_output_outside_inputs(target, inputs)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    _guard_not_in_immutable_run(target)
    if (not isinstance(source_hashes, dict) or not source_hashes
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source_hashes.values())):
        raise ValueError("explicit source code SHA-256 map is required")
    if resolved_config is not None and not isinstance(resolved_config, dict):
        raise ValueError("resolved_config must be a mapping")

    aq, ds = load_evaluation_outputs(inputs[0]), load_evaluation_outputs(inputs[1])
    _check_paired_outputs(aq, ds)
    random = load_random_control_summary(inputs[2])
    bundle_hash = aq["manifest"]["scientific_input_pins"]["bundle_manifest_sha256"]
    if random["manifest"]["scientific_input_pins"]["bundle_manifest_sha256"] != bundle_hash:
        raise ValueError("random control and learned models use different evaluation bundles")
    p = protocol or load_protocol()
    # Fail on missing fonts before creating a presentation run or exporting files.
    apply_style(p)
    config = {
        **(resolved_config or {}),
        "anchor_q_output": str(inputs[0]),
        "dimsim_output": str(inputs[1]),
        "random_control_output": str(inputs[2]),
        "source_digest": source_digest,
        "style_sha256": p.style_hash,
        "registry_sha256": p.registry_hash,
        "formats": list(p.style["export"]["formats"]),
        "png_dpi": p.style["export"]["dpi"],
    }
    pins = {
        "evaluation_bundle_manifest_sha256": bundle_hash,
        "anchor_q_manifest_sha256": file_sha256(inputs[0] / "manifest.json"),
        "dimsim_manifest_sha256": file_sha256(inputs[1] / "manifest.json"),
        "random_control_manifest_sha256": file_sha256(inputs[2] / "manifest.json"),
        "model_ids": aq["manifest"]["scientific_input_pins"]["model_ids"],
        "seeds": aq["manifest"]["scientific_input_pins"]["seeds"],
    }
    with tempfile.TemporaryDirectory(prefix="evaluation-tables-") as directory:
        stage = Path(directory)
        table_paths = render_evaluation_tables(aq["tables"], ds["tables"], stage / "tables")
        expected = {"config.json", "data/random_control_summary.json", "diagnostics/summary.json"}
        expected.update(f"tables/{path.name}" for path in table_paths)
        expected.update(f"figures/{stem}.{suffix}" for stem in FIGURE_STEMS
                        for suffix in ("png", "eps", "metadata.json"))
        with _evaluation_run(
            project_root, "contextual-evaluation-report", run_id,
            input_roots=inputs, config=config, source_hashes=source_hashes,
            expected_files=expected, scientific_pins=pins,
            profile="contextual_evaluation_report", figure_exports_on_complete="complete",
        ) as root:
            for path in table_paths:
                shutil.copyfile(path, root / "tables" / path.name)
            _write(root / "data/random_control_summary.json", random["report"])
            render_evaluation(inputs[0], inputs[1], root / "figures", source_digest, protocol=p)
            _write(root / "diagnostics/summary.json", {
                "scope": "presentation", "figure_count": len(FIGURE_STEMS),
                "table_count": len(table_paths), "evaluation_bundle_manifest_sha256": bundle_hash,
            })
    return target
