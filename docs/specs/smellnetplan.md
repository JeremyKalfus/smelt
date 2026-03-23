# A Ruthless 1‑Week Research Strategy to Beat SCENTFORMER on SmellNet‑Base

## Executive summary

You do **not** win this sprint by inventing a fancy architecture. You win by (a) matching the benchmark *exactly*, (b) avoiding leakage, (c) exploiting what the authors already told you works (p=25 temporal differencing + window size 100 + GC‑MS supervision), and (d) swapping in a more data-efficient, drift-tolerant time‑series backbone that trains cleanly on modest compute.

The highest-probability, one-shot plan is:

Build **one** strong SMELLNET‑BASE model around **(raw + p=25 temporal differences concatenated as channels)**, with **a proven time‑series CNN backbone (InceptionTime‑style)**, plus **GC‑MS contrastive pretraining (as the “chemistry prior”)**, plus **aggressive anti-drift normalization/augmentation**. This leverages (1) the SmellNet paper’s best settings and (2) the existing SmellNet codebase functions for sliding windows, differencing (periods=25), FFT high-pass, and contrastive GC‑MS pairing. citeturn8view0turn48view1turn37view3turn48view0

Your “paper sprint” credibility comes from a **minimal ablation ladder**: baseline SCENTFORMER setting → your backbone swap → add GC‑MS contrastive → add anti-drift “hardening” (FFT / augmentations) → (optional) day-robustness analysis.

## SmellNet and SCENTFORMER: what you are actually trying to beat

SmellNet (ICLR 2026 OpenReview) is described as ~828k time-series points across **50 base substances** and **43 mixtures**, with **68 hours** of data. citeturn8view0turn14view0 The paper also frames SCENTFORMER as a Transformer that combines **temporal differencing** and **sliding-window augmentation** for smell sensor time series. citeturn8view0turn14view0

On SMELLNET‑BASE, OpenReview’s abstract reports **63.3% Top‑1** *with GC‑MS supervision*. citeturn8view0 The arXiv abstract for the same work also reports a **58.5% Top‑1** number (without that “with GC‑MS supervision” qualifier in the abstract), which strongly implies there are multiple regimes/settings being summarized (e.g., sensor-only vs additional supervision). citeturn14view0 Either way, the “beat SCENTFORMER credibly” bar you set is anchored to the **63.3% GC‑MS-supervised** result, and you specifically want to target SMELLNET‑BASE first (correct choice, because mixture recipe prediction is a harder generalization problem and is explicitly a separate benchmark). citeturn8view0turn14view0

Key operational fact: the public SmellNet repo’s training script already supports (a) **window_size=100**, (b) **stride defaults**, (c) **temporal differencing** via `diff_data_like(..., periods=spec.gradient)` where you can set gradient=25, and (d) **contrastive mode** pairing each window with a GC‑MS vector using `create_pair_data(...)`. citeturn48view0turn48view1turn44view2 That is your leverage: you are not starting from scratch; you are reusing their pipeline and swapping the model + training recipe.

## Brutal critique of your “Dual‑View Drift‑Aware ConvTransformer”

I’m going to be blunt and optimize for **probability of a win in 7 days**, not conceptual completeness.

### Keep these parts (high probability per unit effort)

Your instinct to use **p=25 temporal differencing** and **window size 100** is aligned with the strongest reported SCENTFORMER settings you cited and with the codebase’s built-in differencing pathway. citeturn8view0turn48view1turn37view3

Your insistence on **not using a giant Transformer** is correct: on small windows (100 steps) and limited compute, plain Transformers often overfit or underperform stronger CNN/TCN inductive biases unless carefully tuned. A time-series CNN backbone is a better “week sprint” bet, and InceptionTime is a known strong baseline for time series classification. citeturn45search0

Your focus on **SMELLNET‑BASE first** is strategically correct and is supported by the benchmark framing (base classification vs mixture distribution prediction are different tasks; the paper highlights mixture performance separately). citeturn8view0turn14view0

### Cut these parts (low ROI / high risk in a week)

**Two parallel branches (raw branch + diff branch)**: unnecessary complexity. If you want “dual view,” do **channel concatenation** (raw and diff stacked into a single tensor) or, even cheaper, a **two-model logits ensemble** later. Parallel branches buy you little and cost you debugging time.

**Auxiliary 5‑class category head**: likely low value. Category supervision is extremely coarse compared to the 50‑way head, and it can easily become a distraction unless you need it for analysis. If you want category analysis, compute it post hoc from the 50‑way confusion matrix (the repo already supports per-category accuracy in evaluation). citeturn43view1

**Day-classifier with gradient reversal (“day adversarial”)**: this is tempting, but it is high-risk for your constraints because it requires rock-solid “day ID” extraction with **zero leakage** and nontrivial training stability. Domain-adversarial training is real and uses a gradient reversal layer, but it is also easy to implement *wrong* and waste days. citeturn45search1 Do not put this into the core one-shot model unless you have day labels cleanly available and you already match baseline performance.

**Supervised contrastive / prototype losses**: potentially helpful, but it adds another training regime (augment-pairs, temperature tuning, batch composition sensitivity). Supervised contrastive can outperform cross-entropy and help robustness, but in a one-week sprint, this is often a rabbit hole unless you have a strong contrastive infrastructure already. citeturn45search3

### Replace your weak choices with better “week sprint” replacements

Replace “ConvTransformer + multiple aux heads” with:

1) **A strong time-series CNN backbone** (InceptionTime-style multi-kernel Inception blocks), because it is data-efficient for time-series patterns and simple to train end-to-end. citeturn45search0  
2) **GC‑MS supervision as contrastive pretraining**, because SCENTFORMER’s best number is explicitly *with GC‑MS supervision* and the codebase already supports contrastive pairing and training loops. citeturn8view0turn48view0  
3) **Anti-drift hardening** via the existing pipeline knobs: subtract-first-row preprocessing, optional high-pass FFT, window standardization fit on training windows, plus light noise + feature dropout augmentations also present in the codebase. citeturn44view2turn48view0turn37view3

That combo is the best “ruthless” risk-adjusted bet: it targets SCENTFORMER’s strengths (diff + windows + GC‑MS supervision) and attacks generalization fragility with simple, reliable regularization.

## Final recommended model: FIT‑GCMS (Fused InceptionTime + GC‑MS pretraining)

This is your **main bet**. It is deliberately engineered for: (1) fast implementation from the SmellNet codebase, (2) strong performance at window length 100, (3) compatibility with GC‑MS supervision, and (4) robustness hardening without exotic domain adaptation machinery.

### Exact input representation

Let the raw per-recording sensor matrix be **X_raw ∈ R^(T×C)** (T timesteps, C sensor channels).

1) **Baseline subtraction (per recording):**  
   Use the codebase’s approach: subtract the first row from every row in the CSV to normalize relative to ambient baseline. This is already implemented in `load_sensor_data()` where `df = df - df.iloc[0]`. citeturn44view2

2) **Temporal differencing (p=25):**  
   Compute `X_diff = diff(X_raw, periods=25)` (the repo uses `df.diff(periods=periods).iloc[periods:]`). citeturn37view3turn48view1

3) **Raw alignment to diff length:**  
   Crop the raw sequence to match the diff sequence length: `X_raw_crop = X_raw[25:, :]`.

4) **Channel fusion (“dual view” without branches):**  
   Concatenate along channels:  
   **X_fused = concat(X_raw_crop, X_diff) ∈ R^((T-25)×(2C))**.

5) **Sliding windows:**  
   Use window size **w=100**. Adopt stride **50** initially (the loader’s default is stride=50 when window_size=100 in `make_sliding_window_dataset`, and `run.py` defaults stride to window_size//2). citeturn37view3turn48view0  
   This yields windows **W ∈ R^(100×2C)** with the ingredient label y ∈ {1..50}.

Pragmatic note: if you’re killing for peak accuracy and you have compute, you can reduce stride (e.g., 25) to generate more windows, but correlated windows can inflate train accuracy and not help test robustness. Start with 50.

### Preprocessing and normalization

Use only what you can implement correctly and fast:

1) **Column filtering:**  
   The repo’s `run.py` calls `load_sensor_data(..., removed_filtered_columns=[Benzene, Temperature, Pressure, Humidity, Gas_Resistance, Altitude])`. Keep this constant for baseline comparability until you have a win; changing sensor channels is a silent confound. citeturn48view1

2) **High-pass FFT (optional but high-probability for drift):**  
   `run.py` supports `highpass_fft_batch(...)` applied after windowing. This explicitly removes low-frequency components below a cutoff and is the most “mechanically reliable” drift removal knob you have. citeturn37view3turn48view0  
   For w=100, start with a conservative cutoff (e.g., 0.05 in their helper signature) and **treat it as a binary ablation**, not a search.

3) **Standardization:**  
   Use the built-in “fit standardizer on training windows only” then apply to train and test (`fit_standardizer_from_windows` / `apply_standardizer` in `run.py`). This is a critical anti-leakage / anti-shift move and is already wired. citeturn48view0

### Augmentations (simple, targeted, and already supported)

In the classification training loop, enable:

- **Gaussian noise injection** on windows (small σ). The repo provides `apply_noise_injection(X, noise_scale=...)`. citeturn37view3turn44view2  
- **Random feature dropout** across channels (drop a fraction of channels). The repo provides `apply_random_feature_dropout(X, dropout_fraction=...)`. citeturn37view3turn44view2  

Do **not** do fancy time warping, reversing, or learned augment pipelines in week 1; smell dynamics are not symmetric and you’ll waste time validating augmentation realism.

### Architecture (block-by-block, with dimensions)

Backbone: **InceptionTime-style 1D CNN** (single stream).

Why: InceptionTime was proposed as a scalable, high-accuracy time-series classification architecture, competitive with strong non-deep baselines and designed to learn multi-scale patterns via parallel convolutions. citeturn45search0

**Input:** batch tensor `B × T × F`, where `T=100`, `F=2C` channels.

You will implement the backbone in the style of InceptionTime (multi-kernel Inception modules, residual shortcuts). Exact spec:

- **Reformat:** `B × F × T` (channels first for Conv1D).

- **Stem bottleneck (optional but recommended):**  
  `1×1 Conv` projecting `F → 64`.

- **Inception Module (repeat M times):** each module outputs **128 channels**.
  - Bottleneck: `1×1 Conv` 64→32  
  - Branch A: `Conv1D(k=3)` 32→32  
  - Branch B: `Conv1D(k=5)` 32→32  
  - Branch C: `Conv1D(k=9)` 32→32  
  - Branch D: `MaxPool1D(k=3)` then `1×1 Conv` 64→32  
  - Concat branches: 32×4 = 128 channels  
  - Norm + activation: use **GroupNorm** (not BatchNorm) + GELU

- **Residual pattern:**  
  After every 3 Inception modules, add a residual shortcut from the block input to output (use a `1×1 Conv` to match channel dims if needed).

- **Temporal pooling:**  
  Use **global average pooling** over time to get `B × 128`.  
  (If time allows, add a lightweight attention pooling, but do not start there.)

- **Embedding projection head:**  
  MLP: `128 → 256 → 256`, with dropout=0.2 and LayerNorm on the final embedding.

- **Classifier head:**  
  Linear `256 → 50`.

This is intentionally small enough to train quickly on your GTX 1650 / Colab, and strong enough to plausibly outperform the repo’s vanilla CNN or Transformer baselines.

### Heads and losses

You will use **two training phases** (this is key to beating the “with GC‑MS supervision” number without making your classifier training unstable):

#### Phase A: GC‑MS contrastive pretraining (chemistry prior)

Goal: learn a smell-window embedding that aligns with an ingredient’s GC‑MS vector.

- **GC‑MS encoder:** use the repo’s `GCMSMLPEncoder` (outputs 256-dim). citeturn39view3  
- **Sensor encoder:** your Inception backbone up to the 256-dim embedding.
- **Data pairing:** use `create_pair_data(...)` to pair each smell window with the GC‑MS vector indexed by its label. citeturn37view3turn48view3  
- **Batch sampler:** use `UniqueGCMSampler` so a batch contains unique GC‑MS targets (stabilizes contrastive training). citeturn35view0turn48view3  
- **Loss:** cross-modal contrastive loss (InfoNCE style) as implemented in the repo training loop. citeturn41view3turn40view0  
- **Output:** save sensor encoder weights.

Rationale: Your opponent’s best number is explicitly *with GC‑MS supervision*. citeturn8view0 If you ignore GC‑MS entirely, you are betting your architecture alone can beat an architecture + extra supervision. That’s lower probability in a one-week sprint.

#### Phase B: Supervised classification fine-tuning

- Initialize sensor encoder from Phase A.
- Attach the 50-way classifier head.
- Train with **cross entropy**.  
- Optional “safe” improvement: **mixup** on window tensors. Mixup is a simple regularization method that forms convex combinations of examples and labels and often improves generalization. citeturn45search2

### Optimizer, schedule, regularization

The repo uses Adam with no scheduler by default. citeturn41view1 That is fine for baseline reproduction, but for a “win attempt” you should upgrade the recipe slightly in a controlled way:

- **Optimizer:** AdamW (or Adam if you refuse to change code paths).  
- **LR:** start 3e‑4 for fine-tuning, 1e‑3 for contrastive pretraining (contrastive often tolerates higher LR; the repo uses 1e‑3 in `contrastive_train`). citeturn41view1  
- **Weight decay:** 1e‑2 (classification), 1e‑4 (contrastive).  
- **Gradient clipping:** keep ~1.0 (the training loop already supports grad clipping). citeturn41view1  
- **Dropout:** 0.2 in embedding/classifier head; 0.1 in backbone modules.
- **Early stopping:** only if you create a clean internal validation split by **recording**, not by window.

### Validation strategy (the part that makes your result “real”)

You need two evaluation tracks:

1) **Primary claim track: official SMELLNET‑BASE test split**  
   Use the repo’s `offline_training` vs `offline_testing` separation (or the updated split in the current SmellNet codebase you have locally). Never randomly split windows across train/test; generate windows *after* splitting by recording directory. The repo’s run loop does exactly this separation (it builds windows separately from `train_data` and `test_data`). citeturn48view0turn44view2

2) **Robustness track: leave-one-day-out (LODO)**  
   You told me SCENTFORMER’s LODO mean suggests day/domain shift is still a major weakness. Treat LODO as your “robustness win condition.”  
   Implementation note: do not attempt LODO until your primary split training is stable and matches baseline. LODO is where amateurs accidentally leak metadata.

### Expected failure modes (so you can debug fast)

- **Leakage via window splitting:** If you build a giant window array then do a random split, you will get fake-high accuracy. Don’t. Split by recording/day first. The repo’s structure helps you avoid this if you follow it. citeturn48view0  
- **Correlated windows overfitting:** stride too small creates near-duplicate windows and can inflate training metrics without improving test. Start at stride=50 for w=100. citeturn48view0  
- **Normalization mismatch:** fitting scalers on train+test together is silent leakage. Use train-only fit (the repo does this unless `--no-standardize` is set). citeturn48view0  
- **Contrastive pretraining collapse:** if batches contain duplicated GC‑MS vectors, InfoNCE becomes less informative. Use the provided `UniqueGCMSampler`. citeturn35view0turn48view3  
- **FFT cutoff harming signal:** high-pass FFT can remove useful low-frequency smell signatures. Treat FFT as a single on/off ablation; don’t tune it endlessly. citeturn37view3turn48view0

### Why this has a real chance to beat SCENTFORMER on SmellNet‑Base

This approach stacks three high-probability advantages:

- **Matches the opponent’s strongest knobs** (windowed time-series modeling + p=25 differencing + sliding window generation). citeturn8view0turn48view1turn37view3  
- **Uses the same kind of extra supervision that boosts their best number** (GC‑MS supervision), but in a contrastive representation-learning form that is already supported by the repo’s training pathway. citeturn8view0turn48view0  
- **Uses a backbone class known to be strong for time-series classification** (InceptionTime-style multi-scale CNN), which often beats naïve Transformers on short-to-medium time-series windows when training/data are imperfect. citeturn45search0  

If you beat 63.3% Top‑1 on the base task with this, reviewers will believe it because (a) you didn’t change the benchmark, (b) you used the published GC‑MS side information transparently, and (c) you provide clean ablations.

## Backup model: “CNN‑GCMS hardened baseline”

If the InceptionTime implementation or training becomes a time sink, you need a fallback that is already wired.

**Backup = repo’s existing `cnn` model + your preprocessing recipe + GC‑MS contrastive pretraining.**

Concrete spec:

- Input: **diff-only** (gradient=25) OR **raw-only** (gradient=0); do not attempt fused channels in the backup. citeturn48view1turn48view0  
- Window: w=100, stride=50. citeturn48view0  
- Standardize train-only. citeturn48view0  
- Optional FFT high-pass (binary ablation). citeturn48view0turn37view3  
- Phase A: contrastive pretrain (repo’s `contrastive` mode). citeturn48view0turn41view3  
- Phase B: supervised classification fine-tune.

Why this is a good backup: it’s mostly “turn knobs” in `run.py` (model choice, gradient, window size, contrastive on/off, fft on/off). citeturn48view0turn48view1

## A strict 1‑week execution plan

Your day-by-day plan must be **binary**: reproduce → implement → ablate → write. No wandering.

### Day 1: Make the benchmark run and lock down leakage rules

Deliverables by end of day:

- Data downloaded and the official SMELLNET‑BASE pipeline runs end-to-end (train + evaluate).  
- You can run at least one baseline configuration with `run.py` and get an accuracy number (even if low). citeturn48view0  
- You write down the “anti-leakage commandments” in your notes:
  - split by recording folder, not by window  
  - train-only scaling  
  - keep window size/stride fixed for baselines  
  - log seeds and exact CLI args

First run to reproduce (minimum viable baseline):

- `model=transformer`, `gradient=25`, `window_size=100`, `contrastive=on` (to mimic “GC‑MS supervision”), `fft=off`, `standardize=on`. citeturn48view0turn48view1turn8view0  

Continue vs pivot threshold:

- If you cannot get stable training and a non-trivial accuracy by end of Day 1, you stop everything and debug data loading, shapes, and label encoder alignment (the repo explicitly aligns labels to GC‑MS CSV via the LabelEncoder). citeturn48view1turn48view0

### Day 2: Reproduce a credible SCENTFORMER-like baseline in your environment

Goal: you need a baseline number you trust *on your machine*, not the paper.

Run the shortest credible sweep:

- Transformer, w=100, gradient in {0,25}, contrastive in {off,on}. citeturn48view0turn48view1  
- Keep epochs modest (e.g., 30) just to see ranking; increase later.

Decision threshold:

- If contrastive=on does **not** improve over off in your environment (on the same settings), your GC‑MS alignment path is broken (label mismatch, sampler bug, or dataset mismatch). Fix this immediately because your main bet depends on it. citeturn8view0turn48view3

### Day 3: Implement FIT‑GCMS backbone (InceptionTime-style) and hit “baseline parity” quickly

Deliverables:

- New model class added (e.g., `inception`) with `forward()` and `forward_features()`.
- `run.py` recognizes it as a MODEL_CHOICE and can train/evaluate.

Run:

- Without contrastive first: `model=inception`, gradient=25, w=100, standardize=on, fft=off. citeturn48view0turn48view1  
- Your goal today is not SOTA; it’s “does this beat the repo CNN and approach Transformer.”

Continue vs pivot threshold:

- If your Inception model is **worse than the repo CNN** by >2 points after basic tuning (LR, epochs), stop improving the architecture and revert to the backup path. There is no time to debug a fancy backbone that isn’t obviously superior.

### Day 4: Add GC‑MS contrastive pretraining to FIT‑GCMS

Deliverables:

- Two-stage pipeline works:
  1) contrastive pretrain sensor encoder (inception) + GCMSMLPEncoder  
  2) fine-tune classifier head

Run the critical comparisons (these will become your paper ablation table):

- FIT‑GCMS **without** contrastive pretrain  
- FIT‑GCMS **with** contrastive pretrain  
(same seeds, same window/gradient)

Decision threshold:

- If the contrastive pretrain does not help at all (≤ +0.5%), you still keep it if it helps robustness later, but you stop investing time in tuning temperature/samplers unless something is clearly broken. citeturn41view3turn35view0

### Day 5: Hardening for day/domain shift (only “safe” knobs)

Do not add gradient reversal today. Do knobs you can trust.

Binary ablations:

1) FFT high-pass on/off (same model/seed). citeturn48view0turn37view3  
2) Noise injection on/off. citeturn44view2turn41view1  
3) Feature dropout on/off. citeturn44view2turn41view1  

Pick the best stable configuration and lock it.

Decision threshold:

- If FFT helps the robustness proxy you choose (see Day 6) without killing in-distribution accuracy, keep it. If it hurts, drop it and move on—no tuning loops.

### Day 6: Robustness evaluation and minimal additional proof

Today you earn the “credible” part.

You should run some form of “temporal robustness” evaluation, even if it’s not identical to the paper’s LODO:

- If your dataset/codebase includes explicit day partitions: run true leave-one-day-out.
- If not: approximate “day as session/file” and hold out entire recording files as pseudo-domains.

Your claim is not “we solved domain shift”; it is “we materially improved robustness compared to SCENTFORMER-like baseline settings.”

### Day 7: Paper sprint assembly (results, ablations, writing)

You write the paper around a single story:

- “Replacing the Transformer backbone with a multi-scale time-series CNN, while preserving the benchmark’s temporal differencing and adding GC‑MS contrastive alignment, yields higher Top‑1 on SMELLNET‑BASE and improves robustness to temporal shift.”

You do not add experiments today unless:
- a single missing ablation is needed to defend the claim.

## Ablation order optimized for speed and evidentiary value

This is the minimum set that makes your paper defensible:

1) **Baseline (SCENTFORMER-like):** Transformer, w=100, gradient=25, contrastive=on. citeturn48view0turn48view1turn8view0  
2) **Backbone swap only:** Inception backbone, same preprocessing, contrastive=on.  
3) **GC‑MS contribution:** Inception with contrastive=off vs on. citeturn48view0turn41view3  
4) **Anti-drift knob:** FFT off vs on, holding everything else fixed. citeturn48view0turn37view3  
5) **Representation justification:** fused channels vs diff-only (or raw-only). This shows your “dual view” is real value or that diff alone suffices. citeturn48view1turn37view3  

Stop there. Anything beyond this is a trap unless you already won.

## What not to do in this sprint

These are the classic week-long failure modes:

- **Bigger plain Transformer:** scaling depth/width without careful regularization and domain-shift handling is a low-probability win and high-probability overfit. Your dataset windows are short (100) and highly correlated; brute scaling is not where you get gains. citeturn8view0turn48view0  
- **Mixture-first:** the benchmark explicitly separates base classification from mixture distribution prediction, and the paper reports mixture results separately. You already noted mixture generalization is weaker; do not pick the harder task first. citeturn8view0turn14view0  
- **Overcomplicated “multimodal” pipelines:** do not build a full multimodal fusion model. Use GC‑MS only as **contrastive supervision** (representation alignment), because that’s already implemented and mirrors the “with GC‑MS supervision” regime you’re trying to beat. citeturn8view0turn48view3  
- **Giant hyperparameter search:** your win comes from 2–3 decisive interventions, not Bayesian optimization.  
- **Self-supervised rabbit holes:** supervised contrastive is real, but it is a second project and will eat your week. citeturn45search3  
- **Day-adversarial GRL as a first-class dependency:** domain-adversarial training is legitimate, but it is easy to implement incorrectly, and it requires clean domain labels. Put it behind a “only if we already win and need robustness” gate. citeturn45search1  

## The exact paper claim if results work

If you win, your claim should be narrow, testable, and benchmark-aligned:

> On SMELLNET‑BASE, replacing SCENTFORMER’s Transformer backbone with a fused multi-scale CNN (InceptionTime‑style) while preserving temporal differencing (p=25) and adding GC‑MS contrastive alignment improves Top‑1 accuracy over the best reported SCENTFORMER setting and increases robustness under temporal shift evaluation.

You explicitly report:

- the exact window size and differencing period (w=100, p=25). citeturn8view0turn48view1  
- whether GC‑MS supervision is used (contrastive pretrain on/off). citeturn8view0turn48view0  
- the evaluation split (official base test split and your robustness protocol).

## A concrete decision tree for the sprint

Use this to avoid “wandering”:

- **If you cannot reproduce a stable baseline run by end of Day 2:**  
  Stop architecture work. Fix data, scaling, label encoder alignment, and leakage issues. citeturn48view0turn48view1  

- **If Inception backbone (no contrastive) is not at least competitive with Transformer by Day 3:**  
  Pivot to the **backup** (repo CNN + contrastive + hardening). You are not here to admire your own architecture.

- **If contrastive pretraining does not improve accuracy but improves robustness:**  
  Keep it and frame it as a robustness/representation benefit (still paper-worthy). citeturn41view3turn45search3  

- **If FFT improves robustness but hurts Top‑1 slightly (≤1 point):**  
  Keep FFT; in papers, robustness wins are often worth a small in-distribution tradeoff. citeturn48view0turn37view3  

- **If by Day 5 you are not above your best baseline on the official split:**  
  You stop adding features and instead try the simplest high-probability booster: **two-model ensemble** (raw-only + diff-only logits average). It’s ugly, but it’s fast and often wins.

- **Only if you’re already winning on the official split but losing badly on robustness:**  
  Consider day-adversarial GRL as a late-stage experiment, because it is an additional moving part, not a core dependency. citeturn45search1