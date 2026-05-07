# Contributing

This release package is intentionally small. Changes should preserve the public
method set:

- `flop_obs`
- `flop_envwise`
- `i_flop_envwise`

Before submitting changes, run:

```bash
python -m pytest -q
cd rust/iflop_native
cargo test
```

Do not commit generated artifacts such as `target/`, `.pytest_cache/`,
`__pycache__/`, local benchmark outputs, or OS metadata files.
