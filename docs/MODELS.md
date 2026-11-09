# Models and training

Canon4 is the current realtime JPEG repair default, not the entire model family.
The `canon4` artifact name does not certify compliance with imazen-26's published
ID split. See [split authorities](SPLIT_DEFINITIONS.md) for explicit definitions.
The runtime accepts caller-loaded weights. Its current automatic repair router
is calibrated for canon4; another model needs its own routing evaluation.

## Model family

| Model | Purpose | Status |
|---|---|---|
| `dejpeg_rt24i_canon4` | Compact ×1 generalist, 43,203 parameters | Current default; committed f16 weights and goldens |
| `dejpeg_rt24g` | Previous realtime ×1 model | Committed weights retained for comparisons |
| `dejpeg_rt24h_canon` | imazen-26-source retrain using turbo and zenjpeg | Superseded by canon4; local research artifact, not committed |
| `dejpeg_rt24z_zen` | zenjpeg specialist | Not adopted; small median advantages did not justify its mean losses and tails; local artifact |
| `dejpeg7_graphics` | Larger ×1 quality model, 595,459 parameters | Historical model; weights are not included in this checkout |
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

## Quality-model origin and current evidence

`dejpeg7_graphics` is a zensr fine-tune with external initialization, not a
from-scratch model. The retained preview `MODEL-CATALOG.json` records this
parent chain:

```text
2xNomosUni_compact_multijpg_ldl
  → dejpeg_1x → dejpeg2_off → dejpeg2b_off → dejpeg3_off
  → dejpeg4_policy → dejpeg7_graphics
```

The original [NomosUni model card](https://huggingface.co/Phips/2xNomosUni_compact_multijpg_ldl)
credits Philip Hofmann, gives a January 11, 2024 release date and lists
CC BY 4.0. It uses SRVGGNetCompact for ×2 upscaling. zensr's
[initialization loader](../tools/train_people.py) loads matching `body.*`
tensors by key and shape, allowing reuse of a trunk with a different output
scale. The retained metadata names the parents but leaves trainer/dataset
details unknown; this chain is not a complete reproducible training recipe.
The deployed ×1 model has 18 convolutions, 64 hidden channels and 595,459
parameters, stored in 1,190,918 bytes of f16 weights.

The [November 8 precision comparison](../benchmarks/quality_int8_2026-11-08.md)
also measures the native float model with the deployed f16-expanded weights.
The [quality-by-category tables](../benchmarks/quality_by_category_2026-11-08.md)
break out gains, harm and worst losses by nominal JPEG quality, category,
encoder and chroma sampling. Each of the 17 categories has only one source
image; repeated JPEG encodings do not establish category-wide performance.
On 2,720 primary JPEG cases from 17 origins at long side 512, forced repair
with guard/coefficient projection gives **+2.86600 origin-balanced mean
SSIMULACRA2**, **257/2,720 harm cases (9.45%)**, and **−6.62849 worst gain**
against decoded JPEG. Routing and the near-pristine gate are off. These are
reused inputs, not a fresh generalization certification or a comparison to
modern external models. The earlier audit's corrupted-target finding remains
a lineage limitation; its impact on these weights has not been isolated.

## Research after the project's first commit

The first repository commit is `cd1b9f4` (July 21, 2026). Primary-source review
on November 8 found these later papers worth evaluating:

| Work | Method and relevance | Limits of this comparison |
|---|---|---|
| [Wavelet-domain MeanFlow JPEG restoration](https://arxiv.org/abs/2608.28730), August 28 | Invertible Haar transform and residual generation in one or two network evaluations | The [public inference package](https://huggingface.co/asandeistefan/jpeg-restore-meanflows) is available; our count is 65,323,201 parameters. See the [external-model comparison](research/EXTERNAL_MODELS_2026-11-09.md). |
| [PixelIR](https://arxiv.org/abs/2608.30782), August 31 | Separates faithful reconstruction from generated detail; useful for an explicit fidelity/perception control | 32.9M-parameter SR student; generated detail is not a guarantee of recovered source content. |
| [FreeTransformSR v5](https://arxiv.org/abs/2609.05912v5), September 5, revised October 30 | Low-rank learnable transforms with adaptive local processing | Authors report 634K parameters at ×2 and 653K at ×4 on classical SR benchmarks, not our ×1 JPEG repair task. |

These are research candidates. MeanFlow has now run on the reused zensr cohort;
FreeTransformSR and PixelIR still have no matching public checkpoint located.
Paper PSNR/LPIPS or GPU results cannot establish a SSIMULACRA2 or CPU advantage
here. A clean split-respecting reconstruction
baseline followed by a controlled architecture comparison is the recommended
next experiment; it has not been performed by this literature review.

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
   128 × 128 pixels selected through zensr's historical reassigned source
   split, which differs from the published imazen-26 ID split. The final 512
   crops are validation pairs from separate source images within that same
   zensr historical training pool, not imazen-26's published validation bucket. The
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
| Published imazen-26 ID split (`manifests/split_map.tsv` at the revision above) | 1,084 | 658 | 418 |
| zensr historical reassignment (`tools/corpus_split.py`) | 1,151 | 648 | 361 |

The [split helper](../tools/corpus_split.py) groups related images and assigns
each group to its lowest-ID member's bucket. This changes 309 memberships,
including **103 published-imazen-26 validation and 77 published-imazen-26 test images assigned to
zensr's historical training pool**. It also moves 113 published-imazen-26 training images out
of that pool. Therefore canon4's documented selection policy does not preserve
the published imazen-26 held-out boundary.

Python training selects zensr historical `train`; Rust model evaluation reads
zensr historical `validate` plus `test`. The saved zensr map exactly matches regeneration
from the helper and the current corpus snapshot. The split helper, corpus
adapter and training-pool selector match their bytes at trainer revision
`4a33019`; the v4 generator differs only in a documentation-path comment.
This establishes a shared selection policy, not the actual source of every
historical crop. `pairs.tsv` has 48,000 rows but no source-image IDs or crop
coordinates, so those identities cannot be verified from the pair ledger.

The subsequent [exact reconstruction](../benchmarks/canon4_pair_sources_2026-11-07.md)
recovers all 48,000 pair identities and crop coordinates by matching saved
targets and recipe draws, with a separate coordinate-based check. Its training
partition contains 7,985 rows from 162 published-imazen-26 validation/test sources. This improves
the retained dataset's provenance; it does not undo the split-policy violation.

The publication study additionally excludes complete origins touching zensr historical
model training, including stronger document/book grouping. Its separation is
against zensr's historical policy, not proof that model training used only the
published imazen-26 training bucket. See the
[study's data-separation record](../benchmarks/publication_routing_protocol_2026-11-04.md#data-separation).

The history explains how this happened. Commit `69147a0` introduced the
lowest-ID reassignment while stating that the split was no longer defined by
zensr. It treated the corpus's near-duplicate caution as permission to change
membership across published imazen-26 boundaries. Commit `95d4465` expanded grouping
and made evaluation follow the derived split after finding that Python and
Rust disagreed. This repaired agreement between consumers without restoring
the published imazen-26 held-out boundary. The November canon4 run inherited that policy.

Consumer agreement and non-overlap under the derived split do not establish
compliance with the published imazen-26 split. A compliant future training admission
check must require published-imazen-26 `train` membership for each source ID, independently
of grouping; conflicting related sources can be excluded without moving
published-imazen-26 held-out IDs into training. Existing weights and pair archives must
retain their actual policy labels. Changing a manifest cannot retrospectively
remove training exposure, and a compliant retrain would be a separate artifact.

## Artifact distribution

Use GitHub release assets for additional model bundles and `.tar.zst` training
pair archives, keeping manifests, checksums and download instructions in Git.
Each asset must be smaller than 2 GiB under the
[GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
The [uploaded preview records](../benchmarks/preview_assets_2026-11-07.md) give
measured compressed sizes and verified draft assets.

The retained canon4 dataset contains **48,000 pairs**, including the 512-pair
internal validation tail. Its four arrays plus `pairs.tsv` and `meta.json`
total **4,744,799,731 bytes uncompressed** on i265. These are existing artifacts,
not a regenerated dataset. Any archive must distinguish the published imazen-26
source buckets from zensr's historical reassignment, and carry source
revision, existing metadata and the missing-per-pair-identity limitation above;
an archive alone does not repair that provenance gap. The data and model
previews are uploaded as unpublished drafts; see the preview records above.

## Reproduction limits

The run records a command, seeds, dataset metadata, source revision and exported
weight hash. These do not establish a byte-identical retraining recipe: the
model metadata does not bind every training image and encoder binary by hash,
and historical corpus fetches had missing files. Dataset paths in the command
are historical local paths. Regenerating data with current tools is a new run,
not verification of identical original training bytes. Preserve these limits
when describing training provenance or publication reproducibility.

## Canon4 INT8 weight-only experiment

Three paired seeds compare QAT, float fine-tuning and post-training quantization
on the deployed canon4 weights. Each QAT file is 45,408 bytes versus 86,406 bytes
for f16. QAT recovers the mean PTQ loss on the selected cohort, but increases
observed harm; f16 remains the serving default. Activations and inference stay
f32. See the [protocol](CANON4_QAT.md) and
[measured results](../benchmarks/canon4_qat_2026-11-08.md).

## Larger-model PTQ comparison

The retained `dejpeg7_graphics` artifact has 1,190,918 bytes of f16 weights,
expanded to f32 for both native and Tract float execution. Its QLinearConv
INT8 ONNX export is 614,764 bytes; its float ONNX export is 2,384,621 bytes.

On the completed 17-origin, 2,720-case primary comparison, forced repair with
QLinearConv loses 4.71186 mean SSIM2 against native float after guard/coefficient
projection. ConvInteger loses 4.61554. Retaining two original float output
convolutions improves the ConvInteger delta to −3.57714, with a worst loss of
17.40309 against the decoded JPEG. These float layers are insufficient to
recover the native result. A separately trained post-refiner and activation-QAT
for this larger model remain untested.

On the fixed input, ONNX reference and Tract outputs match bit for bit for both
variants with 18 integer convolutions. PReLU and residual addition remain float.
Completed single- and four-thread timing blocks show the tested integer paths
taking longer than native float. The
[complete comparison](../benchmarks/quality_int8_2026-11-08.md) records RGB error
histograms, harm frequency, tails, secondary sizes and timing reliability limits.
[Asset layout and export reproduction](QUALITY_INT8_ASSETS.md).
