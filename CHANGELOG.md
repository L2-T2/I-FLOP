# Changelog

## 1.0.0

- First stable release of the Python-facing I-FLOP package.
- The package, public algorithm entry, method key, CLI, examples, tests, and
  Rust protocol consistently use the name `iflop`.
- Public algorithms are `iflop`, `flop_obs`, and `flop_envwise`.
- Rust backend is the default execution backend for release APIs.
- Python reference backend remains available with `backend="python"` for parity
  checks and debugging.
- R bindings are intentionally not included in this release.
