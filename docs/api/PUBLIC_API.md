# Public API review

Use the [current repair entry-point guide](../API.md) and each crate's source.
The July 2026 text dumps and superseded design proposal were removed because
they omitted the request API, model containers and codec-independent engine.
They remain recoverable from Git history; no current snapshot gate used them.

The product crates are `zensr`, `zensr-zenjpeg`, `zensr-micro` and the separate
`zensr-micro-abi`. `internals` and hidden research items do not become stable
contracts through this inventory. This repository has not published these
repair crates as a crates.io release. Public changes go through review PRs;
a release still requires an explicit API/semver review and release approval.
