"""Versioned artifact inventory for character training runs."""

LEGACY_SCHEMA = 1
COMPACT_SCHEMA = 2
CONTRACT_SOURCE = "training_artifacts.py"
TRAINING_RUN_SCHEMAS = (LEGACY_SCHEMA, COMPACT_SCHEMA)


def required_training_artifacts(seeds, schema):
    per_seed = ("best.pt", "summary.json", "history.json", "events.jsonl")
    if schema == LEGACY_SCHEMA:
        per_seed += ("last.pt", "shared_encoder.pt")
    return {"summary.json"} | {f"seed-{seed}/{name}" for seed in seeds for name in per_seed}
