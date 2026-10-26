# Research orchestration

Research runs use an explicit safe `run_id` and write a single immutable directory under `outputs/<analysis>/<run_id>`. A reserved run starts with a `running` manifest, becomes `failed` on exceptions, and becomes `complete` only after all artifacts have been checksummed. Existing directories are never overwritten.

`wav2vec2_graduality.run_graduality` is the notebook and command-line entry point. Importing this package performs no data loading or computation.
