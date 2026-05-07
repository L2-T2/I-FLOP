# Changelog

## 0.1.0

- Initial release candidate for the final Python-facing I-FLOP package.
- Public API is limited to `flop_obs`, `flop_envwise`, and `i_flop_envwise`.
- Rust native backend is the default execution backend for release APIs.
- Python reference backend remains available with `backend="python"` for parity
  checks and debugging.
- R bindings are intentionally not included in this release.
