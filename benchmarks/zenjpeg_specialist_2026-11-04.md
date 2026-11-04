# zenjpeg-specialist realtime model + 4-encoder canonical generalist — IN PROGRESS 2026-11-04

Question: does a realtime-tier (nf24/nc8, 43k params) model trained only on
zenjpeg output beat the generalist on zenjpeg files, enough to justify a
record/fingerprint-routed second "fast mode" model? And: the 2026-10-30
generalist retrain (`dejpeg_rt24h_canon`) used turbo+zenjpeg only because
i265 lacked mozjpeg/cjpegli; both are installed now (`~/tmp/ati-bin`, copied
from dev / r7900x), so the canonical 4-encoder recipe can run.

Host i265 (RTX 4080). Pipeline: `~/tmp/zensr-zen-pipeline.sh` (PID file
`~/tmp/zensr-zen-pipeline.pid`, log `~/tmp/zensr-zen-pipeline.log`).

| run | data | encoders | out |
|---|---|---|---|
| specialist | `~/tmp/zensr-dejpeg-v4-zen`, 48k pairs | zenjpeg, zenjpeg:noaq, zenjpeg:hybrid, zenjpeg:mozbase, zenjpeg:auto (uniform) | `models/adopted/dejpeg_rt24z_zen` |
| generalist | `~/tmp/zensr-dejpeg-v4-canon4`, 48k pairs | turbo, mozjpeg, jpegli, zenjpeg (canonical mix) | `models/adopted/dejpeg_rt24i_canon4` |

Both: rt24g recipe (`train_people.py 200000 48 3e-4`, QBOOST=3), canonical
corpus pool (1,094 train / 57 val files). Logs `~/tmp/zensr-runs/<out>.log`.

Verdict rule (ROADMAP §0.2): `dejpeg_eval` on the clean refs
(`~/data/zensr/imazen-26-clean-2026-10-30`), same grid as
`zenjpeg_knobs_2026-10-30.md` plus the mozjpeg and jpegli legs, paired per
file via `tools/knob_report.py`. The specialist ships only if it beats the
generalist on the zenjpeg legs by more than the metric floor (0.3 ssim2) on
most files AND the routing signal (record `TableSource`/family, or probe
`Cjpegli*`) is available on the files it would serve.

Record support landed in zensr-zenjpeg the same day (`record.rs`): verified
records override the fingerprint before every policy, so routing to a
specialist can key on `TableSource` + dial rather than the estimate.
