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

## How canon4 was made

The committed [metadata](../models/adopted/dejpeg_rt24i_canon4/meta.json) and
[retraining command](../models/adopted/dejpeg_rt24i_canon4/repro.sh) record the
run at source revision `4a33019`:

1. **Prepare paired data.** The recorded dataset contains 48,000 crops of
   128 × 128 pixels from canonical imazen-26 training content. The final 512
   crops are validation pairs from separate source images. The
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

## Reproduction limits

The run records a command, seeds, dataset metadata, source revision and exported
weight hash. These do not establish a byte-identical retraining recipe: the
model metadata does not bind every training image and encoder binary by hash,
and historical corpus fetches had missing files. Dataset paths in the command
are historical local paths. Regenerating data with current tools is a new run,
not verification of identical original training bytes. Preserve these limits
when describing training provenance or publication reproducibility.
