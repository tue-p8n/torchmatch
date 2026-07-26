# benchmarks/

Per-machine benchmark results contributed by users. Every entry under
`results/<slug>/` is a scrubbed pytest-benchmark JSON plus a
`machine.json` sidecar describing the hardware.

To contribute a run from your machine, see
[Contributing benchmarks](../docs/site/content/5.benchmarks/contributing.md).
The short version:

```bash
uv run python -m torchmatch.bench init-machine   # once per machine
uv run python -m torchmatch.bench collect        # once per release
```

The site picks the data up automatically once your PR merges.
