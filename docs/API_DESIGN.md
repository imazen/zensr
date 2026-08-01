# zensr restoration API — design proposal (2026-08-01, pre-0.1)

Nothing has shipped. This document redesigns the public surface from the
measured results rather than from the order we happened to discover them.
Every design choice below cites the measurement that forces it.

Current shape (zenjpeg PR #191, `restore::restore(data, &model, &opts)`) is an
accretion: a bag of research constants (`high_q_threshold`, `slack_q`,
`slack_abs`, `deblock_policy`) exposed to callers who have no way to reason
about them, an objective (fidelity) hardcoded by omission, and a model chosen
by name. All three are wrong.

---

## 1. What the research actually established

| # | Finding | API consequence |
|---|---|---|
| F1 | **The projection is the big lever, not the model.** No-projection → projected is +6 ssim2 on screen content; the *choice* of slack is worth ~0.3. | The consistency guarantee is the headline feature and must be first-class, not an option flag. |
| F2 | **Correct slack is a function of (encoder family × generation count)**, not quality. Trellis (mozjpeg) violates the box by 9–15 Q at gen1; round-to-nearest by 0.1 Q. Each extra generation adds ~1–2.5 Q. | Slack cannot be a caller-facing number. It is derived from provenance the caller may know and we cannot measure. |
| F2b | **Generation count is not reliably detectable.** Conservative operating point does not exist (false-gen1 = 0 ⇒ gen1-recall = 0). Same-encoder equal-q recompression is invisible (9.4% recall). | Provenance must be an **input**, not an inference. Absence of evidence must never be read as gen1. |
| F3 | **The model hurts on near-pristine input** (invents detail where nothing was destroyed) and helps enormously on damaged input. | "Do nothing" is a legitimate, common outcome and must be expressible in the return type. |
| F4 | **Fidelity metrics may be the wrong objective.** The worst "regression" in our whole grid is the model adding plausible detail into smooth gradients; ssim2 scores that as error. Humans frequently prefer it. Unmeasured — squintly study pending. | The API must not hardcode a fidelity objective. Intent is a caller parameter. |
| F5 | **Content class matters** (graphics vs photo routing measured; low-q graphics wants a different model). | Model selection is a policy decision the library makes, not a name the caller passes. |
| F6 | **Cost varies 16× across tiers** (0.16 vs 2.7 s/MP) for 1.4–3.4× the gain. | Budget is a first-class caller parameter; the library picks the model to fit it. |
| F7 | **Chaining before SR pays only when 4:2:0 or q≲50.** | If we ever expose SR, the chain decision is ours, not the caller's. |
| F8 | **Report granularity hid a real defect for weeks** (cell means vs per-file; contaminated references). | Every decision the library makes must be reportable, and reports must be specific. |

---

## 2. The redesign

### 2.1 Callers declare **intent** and **provenance**; the library owns policy

```rust
/// What the caller is optimising for. There is no default — the honest
/// answer differs per product, and F4 says we cannot pick for them.
#[non_exhaustive]
pub enum Intent {
    /// Maximise measured closeness to the pre-compression original.
    /// Never invents detail. This is what an archival or diff-sensitive
    /// pipeline wants.
    Fidelity,
    /// Maximise perceived quality. May synthesise plausible detail that was
    /// destroyed. Scores *worse* on fidelity metrics by construction (F4).
    Appearance,
    /// Never make any region worse than the plain decode, at the cost of
    /// most of the gain. For blind mass processing where a single bad
    /// output is expensive.
    DoNoHarm,
}

/// What the caller knows about where these bytes came from. We cannot
/// measure this (F2b) and guessing it wrong is unsafe.
#[non_exhaustive]
pub enum Provenance {
    /// Third-party file of unknown history. Conservative slack.
    Unknown,
    /// This process encoded it, once, from an uncompressed source.
    /// Unlocks the strict consistency box (worth +0.3..+0.7 ssim2, F1).
    FreshEncode,
    /// Caller tracked the chain (CDN re-saves, editor round-trips).
    Generations(u8),
}
```

Rationale: `slack_q = 0.15, slack_abs = 1.5` are *our* research constants. A
caller cannot know whether 1.5 is right; they *can* know whether they just
encoded the file. The API asks the question they can answer.

### 2.2 The consistency guarantee is the product, and it is reported

```rust
/// What we can certify about the output.
#[non_exhaustive]
pub enum Consistency {
    /// Every output coefficient lies within the interval the file's own
    /// quantisation tables certify. Re-encoding with those tables
    /// reproduces the input's coefficients. This property is what makes
    /// restoration safe to apply blind (F1).
    Certified { half_width_q: f32, absolute_units: f32 },
    /// Projection could not be applied. `why` is specific: 4:2:2/4:4:0
    /// chroma geometry, CMYK, or caller opt-out.
    NotEnforced { why: NotEnforced },
}
```

No other JPEG restorer offers this. It should be the first thing the docs say,
and callers who need it should be able to *require* it (below).

### 2.3 "Unchanged" is a first-class outcome

```rust
pub enum Outcome {
    /// Policy determined restoration would not help (near-pristine input,
    /// F3) or could not be applied safely. `pixels` is the plain decode.
    Unchanged { pixels: Planar, why: SkipReason },
    /// Restoration applied.
    Restored { pixels: Planar, consistency: Consistency, report: Report },
}
```

This matters practically: a CDN can skip re-encoding entirely when the answer
is `Unchanged`, and F3 says that will be a large fraction of high-quality
traffic. Burying it in a boolean field would hide the most common fast path.

### 2.4 Budget, not model names

```rust
#[non_exhaustive]
pub enum Budget {
    /// ~0.16 s/MP (measured, 12 threads). 84 KB of weights.
    Realtime,
    /// ~2.7 s/MP. 1.16 MB. 1.4–3.4× the gain depending on damage (F6).
    Quality,
    /// Library picks per image, targeting the given per-megapixel budget.
    Adaptive { ms_per_mp: f32 },
}
```

Callers should never write `dejpeg_rt24g`. Model identity is a research
artifact; the tier is the product concept. `Adaptive` is where F5/F6 pay off:
spend the quality tier only where the damage justifies it.

### 2.5 One builder, explicit requirements

```rust
let restorer = Restorer::new(weights)?;          // built once, reusable

let out = restorer
    .restore(&jpeg_bytes)
    .intent(Intent::Fidelity)
    .provenance(Provenance::FreshEncode)         // unlocks the strict box
    .budget(Budget::Realtime)
    .require_consistency(true)                   // else return Unchanged
    .run()?;

match out {
    Outcome::Unchanged { why, .. } => metrics.skipped(why),
    Outcome::Restored { pixels, consistency, report } => { … }
}
```

`require_consistency` exists because some pipelines would rather do nothing
than emit pixels that cannot be certified — that's a policy only the caller
can set, and it's cheap for us to honour.

### 2.6 The report explains the decision, specifically (F8)

```rust
pub struct Report {
    pub encoder_family: EncoderFamily,     // native enum, not a Debug string
    pub estimated_quality: QualityEstimate,
    pub deblock: DeblockChoice,            // which filter and why
    pub model: ModelTier,                  // what actually ran
    pub content_class: Option<ContentClass>,
    pub projection: ProjectionReport,      // clamped fraction, mean change
    pub slack_source: SlackSource,         // Provenance-derived | family default
}
```

Every field answers "why did it do that", which is what we needed and lacked
during the last two weeks of measurement.

---

## 3. What this design deliberately does **not** do

- **No metric is named anywhere in the API.** F4 says our objective function is
  itself under test; an API that says `target_ssim2` would bake in a claim we
  cannot support.
- **No generation detection.** F2b measured that it does not work at a safe
  operating point. Offering `auto_detect_generations()` would be selling a
  9%-recall coin flip as a safety feature.
- **No caller-visible slack numbers.** They are derived from `Provenance` and
  the probed encoder family. If a research caller needs them, that belongs
  behind an `unstable-research` feature, not in the 0.1 surface.
- **No SR in this API.** Chaining is measured (F7) but SR is a separate concern
  with its own scale/geometry questions; mixing them now would freeze a
  coupling we may not want.

---

## 4. Open questions that should gate 0.1

1. **Is `Intent::Appearance` real?** It is currently a hypothesis: our metrics
   penalise invented detail, and we believe humans may not. The squintly study
   (`ROADMAP.md` §4) decides whether this variant ships or is removed. **Do not
   ship an enum variant we cannot justify with data.**
2. **Where is the high-quality cutoff?** Measured crossover moved from q82 to
   above q85 once contaminated references were excluded, and it is single-
   generation only. Needs the gen2 sweep before a constant is frozen.
3. **Does `DoNoHarm` have an implementation?** Today it would mean "strict
   projection + conservative gate", which reduces but does not eliminate
   per-file regressions (25/112 negatives at q85). Either we find a genuine
   never-worse configuration (per-region fallback to the decode?) or we rename
   the variant to something honest.
4. **`Adaptive` budget needs a damage estimator** good enough to route without
   running both tiers. The probe gives quality; content class gives the rest.

---

## 5. Migration from the PR #191 shape

The current `restore()` becomes an internal function. `RestoreOptions`'
research constants move behind `unstable-research`. The feature stays
default-off. No caller exists yet, so there is no compatibility burden — which
is exactly why this is the moment to fix the shape.
