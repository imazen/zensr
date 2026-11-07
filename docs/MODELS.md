# Models and training

Canon4 is the current realtime JPEG repair default, not the entire model family.
The runtime accepts caller-loaded weights. Its current automatic repair router
is calibrated for canon4; another model needs its own routing evaluation.

## Model family

| Model | Purpose | Status |
|---|---|---|
| `dejpeg_rt24i_canon4` | Compact ×1 generalist, 43,203 parameters | Current default; committed f16 weights and goldens |
| `dejpeg_rt24g` | Previous realtime ×1 model | Committed weights retained for comparisons |
| `dejpeg_rt24h_canon` | Canonical-corpus retrain using turbo and zenjpeg | Superseded by canon4; local research artifact, not committed |
| `dejpeg_rt24z_zen` | zenjpeg specialist | Not adopted; small median advantages did not justify its mean losses and tails; local artifact |
| `dejpeg7_graphics` | Larger ×1 quality model, historically reported as 595k parameters | Historical model; weights are not included in this checkout |
| `dejpeg9_gfxycc` | Larger low-quality graphics specialist | Historical candidate; harmful on photographs in measured higher-quality bands; weights not included here |

The [production comparison](../benchmarks/prod_models_2026-11-04.md) records why
canon4 replaced rt24g and why the zenjpeg specialist was rejected. This is a
selection among the realtime retrains, not proof that canon4 beats the larger
quality model. Their recorded scores are from different comparisons.

Separate ×2/×4 SR models and teacher/student distillation experiments also
precede canon4. See the [historical results](research/HISTORICAL_RESULTS.md) and
[research plan](research/PLAN.md). The September
[provenance audit](MODEL-PROVENANCE-AUDIT.md) recorded 47 checkpoints then;
that is a dated inventory, not 47 models distributed in today's checkout.
It also found corrupted targets in the older quality-model fine-tune lineage.
Canon4 was trained from scratch and does not inherit those weights.

## Retained artifact locations

Read-only filesystem checks on **2026-11-07** found the following under each
host's primary checkout, at `models/adopted/`:

| Host | Verified f16 artifacts |
|---|---|
| `i265` | Four model directories: `dejpeg_rt24g`, `dejpeg_rt24h_canon`, `dejpeg_rt24i_canon4`, `dejpeg_rt24z_zen` |
| `dev` | 47 weight files, including `dejpeg7_graphics`, `dejpeg9_gfxycc`, older realtime experiments, teacher/distillation runs and ×2/×4 SR models |

The weight filename is `weights_f16.raw` inside each model directory. The two
larger repair models on `dev` each have **1,190,918 bytes** of f16 weights.
Examples of retained SR models there are `nomosuni_compact_2x`,
`nomosuni_span_2x`, `nomosuni_span_4x`, `general_x4v3`, `general_wdn_x4v3`,
`people_rtc_2x` and `rtc_distill_2x`.

Only rt24g and canon4 weights are tracked by this repository; a fresh clone
does not include the other retained artifacts. These locations establish
presence, not a new quality or provenance validation. Recheck each host's
`models/adopted/*/weights_f16.raw` before relying on this dated inventory.

## How canon4 was made

The committed [metadata](../models/adopted/dejpeg_rt24i_canon4/meta.json) and
[retraining command](../models/adopted/dejpeg_rt24i_canon4/repro.sh) record the
run at source revision `4a33019`:

1. **Prepare paired data.** The recorded dataset contains 48,000 crops of
   128 × 128 pixels selected through zensr's effective imazen-26 training
   split, which differs from the corpus's original split. The final 512
   crops are validation pairs from separate source images within that same
   effective training pool, not the corpus's canonical validation bucket. The
   [generator](../tools/make_dejpeg_data4.py) chooses among turbo, mozjpeg,
   jpegli and zenjpeg, samples integer quality 5–96, and chooses 4:2:0 with
   probability 0.6, otherwise 4:4:4. Five percent of pairs are clean identity
   examples. The input is zenjpeg-decoded RGB; the target is the reference crop.
   Recorded generation-loss augmentation is off.
2. **Train a small residual CNN from scratch.** The
   [trainer](../tools/train_people.py) uses the Compact architecture: an input
   3 × 3 convolution, eight hidden 3 × 3 convolutions, and an output
   convolution, with 24 hidden channels and PReLU activations. Scale is ×1;
   the network predicts a correction added to the input RGB pixels.
3. **Optimize reconstruction.** The recorded run uses 200,000 steps, batch 48,
   initial learning rate 0.0003, AdamW and cosine learning-rate decay. Its loss
   is RGB Charbonnier, a smooth approximation to absolute pixel error.
   `ZENSR_QBOOST=3` adds three extra sampling entries for each high-quality
   (q ≥ 85) or clean training pair. Seeds are torch 7 and NumPy 11. Training
   used CUDA with bfloat16 mixed precision and checkpoints every 5,000 steps.
4. **Export and verify.** Parameters are stored as f16: **43,203 values,
   86,406 bytes**. Goldens are generated after converting weights to f16 and
   back to f32, so they exercise the deployed artifact. Rust inference is
   checked against these goldens and whole-image/tiled parity.
5. **Evaluate the complete repair pipeline and fit routing separately.**
   SSIMULACRA2 measurements select and assess repair policies; SSIMULACRA2 is
   not this network's training loss. JPEG coefficient projection is applied
   after the network. The [shipping record](../benchmarks/ship_rt24i_canon4_2026-11-04.md)
   describes the model comparison and initial calibration; the
   [routing contract](DEFAULT_ROUTING.md) describes today's policy.

Despite the trainer's historical name, `train_people.py`, this run uses the
mixed-content JPEG dataset. The generator also writes severity and uncertainty
channels for conditioning experiments; canon4's network reads RGB only.
It uses reference targets, rather than teacher-output distillation.

## Training and held-out membership

Rechecked on 2026-11-07 against corpus revision
`187fbf338ce08e8e6654db7f04ddae58d5263da2`:

| Split policy | Train | Validate | Test |
|---|---:|---:|---:|
| Canonical imazen-26 | 1,084 | 658 | 418 |
| zensr effective split | 1,151 | 648 | 361 |

The [split helper](../tools/corpus_split.py) groups related images and assigns
each group to its lowest-ID member's bucket. This changes 309 memberships,
including **103 canonical validation and 77 canonical test images assigned to
the effective training pool**. It also moves 113 canonical training images out
of that pool. Therefore canon4's documented selection policy does not preserve
the original canonical held-out boundary.

Python training selects effective `train`; Rust model evaluation reads effective
`validate` plus `test`. The saved effective split exactly matches regeneration
from the helper and the current corpus snapshot. The split helper, corpus
adapter and training-pool selector match their bytes at trainer revision
`4a33019`; the v4 generator differs only in a documentation-path comment.
This establishes a shared selection policy, not the actual source of every
historical crop. `pairs.tsv` has 48,000 rows but no source-image IDs or crop
coordinates, so those identities cannot be verified from the pair ledger.

The publication study additionally excludes complete origins touching effective
model training, including stronger document/book grouping. Its separation is
against the effective policy, not proof that model training used only the raw
canonical training bucket. See the
[study's data-separation record](../benchmarks/publication_routing_protocol_2026-11-04.md#data-separation).

## Artifact distribution

Use GitHub release assets for additional model bundles and `.tar.zst` training
pair archives, keeping manifests, checksums and download instructions in Git.
Each asset must be smaller than 2 GiB under the
[GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
Shard archives if needed; compressed sizes have not been measured.

The retained canon4 dataset contains **48,000 pairs**, including the 512-pair
internal validation tail. Its four arrays plus `pairs.tsv` and `meta.json`
total **4,744,799,731 bytes uncompressed** on i265. These are existing artifacts,
not a regenerated dataset. Any archive must carry the effective split, source
revision, existing metadata and the missing-per-pair-identity limitation above;
an archive alone does not repair that provenance gap. No release assets have
been uploaded as part of this inventory.

## Reproduction limits

The run records a command, seeds, dataset metadata, source revision and exported
weight hash. These do not establish a byte-identical retraining recipe: the
model metadata does not bind every training image and encoder binary by hash,
and historical corpus fetches had missing files. Dataset paths in the command
are historical local paths. Regenerating data with current tools is a new run,
not verification of identical original training bytes. Preserve these limits
when describing training provenance or publication reproducibility.
