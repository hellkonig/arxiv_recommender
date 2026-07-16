# Benchmarks

Local benchmarks for engineering measurements. These scripts are not part of
the installed runtime package.

## Embedding Latency

Measure how long the configured embedding model takes to process representative
title-and-abstract inputs on the current machine:

```bash
uv run python benchmarks/embedding_latency.py \
  --config configs/config.json \
  --sizes 1 10 100 \
  --repeats 3 \
  --warmup 1
```

The benchmark disables the embedding cache so results reflect model inference
latency rather than cache hits.

The output is JSON and includes model configuration, local runtime metadata,
model load time, warmup time, and per-size throughput.

Results are machine-specific. Use them to compare local changes such as
different embedding models, pooling settings, max lengths, or future batch
embedding support.
