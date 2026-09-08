# picker_gain cells, 2026-08-04 — moved off scratch

24,137 rows, 1.9 MB, produced by `crates/zensr-bench/src/bin/picker_gain.rs`
against `/mnt/v/output/clean-picker-corpus-2026-06-26` (the size-diverse picker
corpus, whose 319 leakage-safe origins are pinned in
`eval_split/picker_safe_origins_2026-08-04.txt` — **now superseded by
`..._2026-09-08.txt`: re-measured against the repointed training set, 171 safe
origins and 1,865 renditions, not 319 and 3,452**).

**Still unanalysed.** It was left in `~/tmp`, which is not durable — this move is
the fix, not the analysis. `docs/CORPUS-REPOINT-HANDOFF.md` §8 lists it as work
that does NOT need redoing, so it is worth keeping: the picker corpus never
touched the invalid `/mnt/v/imazen-26` root, and its renditions span `scale36x64`
upward, which is the size diversity the XL corpus lacks entirely (every XL image
is a 512 crop).

| | |
|---|---|
| block storage | `/mnt/v/zensr/picker-gain/2026-08-04/picker_gain.tsv` |
| sha256 | `ef67d40453cf4993b889cefd03def47b94ce5cefc326d5788c816f54d9deb911` |
| rows | 24,137 (+ header) |
| columns | `ref q width height stored_identity recomputed_identity identity_delta restored gain` |
| produced by | `picker_gain.rs`, run 2026-08-04 |

Not committed as bytes: >30 KB, and per the ML-pipeline discipline large generated
data lives in block storage behind a tracked pointer.
