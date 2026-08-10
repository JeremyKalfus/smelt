# Smelt: a protocol-hardened detector for SMELLNET-BASE

Prepared for Dewei Feng · repo: https://github.com/JeremyKalfus/smelt

## 1. Headline

Under the **benchmark-faithful 6-channel setting**, a protocol-hardened detector reaches
**78.0 file-level Top-1** on the official test split, versus **66.0** for the reproduced
sensor-only CNN baseline evaluated the same way — **+12.0 points, exact McNemar p = 0.031**,
and the detector never misses a file the baseline gets right (6–0 discordance).

The previously reported 12-channel number is retained as a clearly separated research
extension, and — importantly — **its additional gain over the 6-channel detector is not
statistically significant** (Section 4). That is a point in favour of the 6-channel release
decision, not against it.

| Setting | Channels | File Top-1 | Top-5 | Macro-F1 |
|---|---|---|---|---|
| Reproduced CNN baseline (file-level) | 6 | 66.0 [52.0, 78.0] | 94.0 | 59.90 |
| **m05b enhanced detector (primary)** | **6** | **78.0 [66.0, 88.0]** | **94.0** | **73.47** |
| m05 enhanced detector (research extension) | 12 | 88.0 [78.0, 96.0] | 100.0 | 84.67 |
| m05 re-run on different hardware (Section 5) | 12 | 86.0 | 98.0 | 82.33 |

95% bootstrap CIs, 10,000 resamples over the 50 official test files.

## 2. Protocol (identical for m05b and m05; only the channel set differs)

- Official `offline_training` / `offline_testing` split, **split by recording file before any
  window generation** (250 train CSVs / 50 test CSVs).
- Diff-only input, `g=25`, `window_size=100`, `stride=50`, train-only standardization.
- **Grouped 5-fold CV** over the official training split: one held-out file per class per fold;
  every training file serves as validation exactly once.
- A 10-model bank (5× CNN seeds, 2× deep temporal ResNet, 2× H-Inception, 1× patch
  transformer). Per-member file aggregator and epoch budget are locked **from CV/OOF evidence
  only**.
- Ensemble members, method and weights selected on CV/OOF only. **No candidate-level
  official-test metric exists anywhere before finalization.**
- Frozen members refit on the full official training split; the official test set is evaluated
  exactly once, at the end.

**Selection calibration.** m05b's CV/OOF estimate was **77.2** Top-1 and the final held-out
test result was **78.0** — a 0.8-point gap, which is the strongest evidence that the selection
procedure is not overfitting the protocol.

Selected m05b ensemble: `weighted_probabilities_all` over all 10 bank members. (The 12-channel
run instead selected a 4-member diversity-greedy subset — with weaker per-model signal at 6
channels, breadth beats selectivity.)

## 3. What the detector adds over the baseline

Both arms below use the **same 6 channels, the same 50 test files, and the same file-level
metric**, so the delta isolates the detector.

| | Value |
|---|---|
| Baseline file Top-1 | 66.0 |
| m05b file Top-1 | 78.0 |
| Delta | **+12.0** [+4.0, +22.0] |
| Files only m05b gets right | 6 |
| Files only baseline gets right | **0** |
| Exact McNemar (two-sided) | **p = 0.031** |

The gains come from grouped file-level validation, validation-locked file aggregation, and
heterogeneous ensembling — not from extra sensors.

## 4. The 12-channel extension, and why it stays a footnote

| Comparison | Delta Top-1 | 95% CI | McNemar p |
|---|---|---|---|
| m05 (12ch, original) − m05b (6ch) | +10.0 | [−2.0, +24.0] | 0.227 |
| m05 (12ch, same machine) − m05b (6ch) | +8.0 | [−4.0, +20.0] | 0.344 |

**Both confidence intervals cross zero and neither test is significant at n = 50.** With 36
files correct under both settings and only 7–8 discordant either way, the honest statement is
that the extra channels are *not demonstrably* better on this benchmark. Note this is an
underpowered comparison rather than a null one: the point estimate stayed positive under two
independent pairings, so a real but modest effect is entirely consistent with the data.

### 4.1 Where the 12-channel signal actually comes from

To identify the mechanism we ran a deliberately weak probe: per-file summary statistics of the
`g=25` differenced signal, nearest-centroid classifier fit on training files only, scored on the
official test files (`results/tables/channel_signal_probe.csv`). Chance = 2.0.

| Channel group | Channels | File Top-1 |
|---|---|---|
| All 12 | 12 | 72.0 |
| **Benchmark 6 + `Gas_Resistance`** | **7** | **70.0** |
| Benchmark 6 | 6 | 54.0 |
| Dropped 6 | 6 | 36.0 |
| **`Gas_Resistance` alone** | **1** | **30.0** |
| Dropped 6 without `Gas_Resistance` | 5 | 28.0 |
| Environmental (Temp/Pressure/Humidity/Altitude) | 4 | 26.0 |
| `Humidity` alone | 1 | 24.0 |
| `NO2` alone (best retained channel) | 1 | 22.0 |
| `Pressure` / `Altitude` alone | 1 | 16.0 |
| `Benzene` alone | 1 | 4.0 |
| `Temperature` alone | 1 | 4.0 |

Three conclusions:

1. **`Gas_Resistance` is the single most informative channel in the dataset** — 30.0 on its own,
   above every benchmark-retained gas channel (best: `NO2` at 22.0). This is not a confound: it
   is the BME680's VOC-sensitive gas reading, i.e. a genuine chemical sensor.
2. **Adding only `Gas_Resistance` to the benchmark set recovers almost the entire 12-channel
   advantage** (54.0 → 70.0, versus 72.0 for all twelve). The other five dropped channels
   contribute little beyond it.
3. **The confound risk narrows to `Pressure`/`Altitude`.** Both sit at 16.0 (8× chance) despite
   having no plausible odor-response mechanism — `Altitude` is derived from `Pressure`, so this
   is one signal counted twice, and it most likely encodes recording-session drift.
   `Temperature` and the broken `Benzene` are at chance, and `Humidity` (24.0) is ambiguous
   since sample evaporation genuinely changes local humidity.

Practical implication: if a future release restores any channel, **`Gas_Resistance` is the one
worth restoring** — high information, mechanistically legitimate, and not implicated in the
fingerprinting concern. Caveat: this probe measures per-file information content under a weak
classifier; it does not prove the deep models exploit the same structure.

## 5. Channel-quality audit (new)

Per-channel scan of the raw 12-column data, both splits
(`results/tables/channel_quality_audit.csv`):

| Channel | Retained | Sentinel values | Flatline train files |
|---|---|---|---|
| **Benzene** | no | **32.4%** (uint32 overflow `4294967295`) | **222 / 250** |
| Humidity | no | 0% | 49 / 250 |
| Gas_Resistance | no | 0% | 49 / 250 |
| Altitude | no | 0% | 10 / 250 |
| Temperature | no | 0% | 7 / 250 |
| Pressure | no | 0% | 3 / 250 |
| Alcohol | **yes** | 0% | **26 / 250** |
| NO2, C2H5OH, VOC, CO, LPG | yes | 0% | 0 / 250 |

Benzene is unambiguously dead — a third of its raw values are the uint32 overflow sentinel and
89% of training files are complete flatlines. This independently confirms the reviewer's
malfunction report. Worth flagging: the **retained** `Alcohol` channel also flatlines in 26
training and 5 test files, so the 6-channel set is not entirely clean either.

## 5b. Paper-informed analyses (new)

Cross-checking against the ICLR camera-ready (arXiv 2506.00239) added four findings:

**(a) Anchors verified.** Our reproduced baselines match the paper's Table 2 exactly
(CNN w=100/p=25: 52.7/85.6/50.5; ScentFormer: 56.1/87.4/55.5), and the paper's own KDE
analysis independently corroborates the probe in §4.1: *"gas resistance sensor is a powerful
discriminator for fruits, vegetables and nuts."* The paper also states the environmental
sensors were included so models can *disentangle* ambient effects — i.e., as covariates, not
odor signal — which matches how we now recommend treating them.

**(b) The dead-BME680 class block.** The paper's App. Table 16 shows nine classes (apple,
lemon, mango, pear, pineapple, pistachios, radish, star anise, strawberry) with identical
frozen environmental readings (Pressure 688.60 hPa, Humidity 100.0, Gas_Resistance 0.0,
std = 0.00) — one BME680 unit was dead for that recording block. This aligns with our
channel-quality audit and means those classes carry a *constant* environmental signature.

**(c) The session-fingerprint concern is real but small — we tested it.** Since each file is
one collection day (the paper's split holds out a day per class), we probed how much class
identity the environmental channels leak (`results/tables/env_barcode_probe.csv`,
nearest-centroid, chance = 2%): raw absolute levels reach 32.0 (ENV4), but the day-held-out
split defeats most of it (test-day pressure ≠ training-day pressure; Pressure alone: 8.0),
and the pipeline's differencing reduces ENV4 to 22.0. The residual leak is mostly the frozen
dead-sensor block from (b). Conclusion: environmental leakage exists but cannot explain the
12-channel gain; the `Gas_Resistance` mechanism from §4.1 remains the primary explanation.

**(d) Online (real-time) split evaluation.** The pre-reorganization dataset includes the
real-time recordings from the arXiv version of the paper (online_nuts: 10 files/10 classes;
online_spices: 13 files/8 classes). We evaluated both frozen ensembles once, 50-way, with no
selection contact (`results/tables/m05b_online_eval_summary.json`):

| Frozen ensemble | Overall | Nuts | Spices |
|---|---|---|---|
| m05b (6ch) | 30.4 | 10.0 | 46.2 |
| m05 (12ch, same-machine refit) | 30.4 | 10.0 | 46.2 |
| arXiv v1 reported models | — | 10.7 | 25.4 |

Two observations, with caveats: the two ensembles score *identically* — whatever the extra
channels contribute offline does not transfer under distribution shift — and the 6-channel
ensemble's spice generalization (46.2) is well above the arXiv-reported 25.4, though the
upstream metric granularity may differ (ours is file-level over 13 recordings; n is small,
and the camera-ready dropped this evaluation). We report it as context, not a head-to-head
claim. Five online files contained 43 empty cells (dropped 1 Hz reads), forward-filled and
counted in the export.

## 6. Reproducibility

- **Cross-release check.** The public dataset was reorganized on 2026-04-13 (12-column
  `data/offline_*` → 6-column `base_data/*`). We verified **all 300 base recordings are
  content-identical on the 6 retained channels** across the two releases (filenames changed;
  matched file-by-file within class). So m05b reproduces exactly from the *current* public
  release; the 12-channel extension requires archived revision `71bcae740b88`.
- **Cross-platform check.** The 12-channel protocol was re-run end-to-end on entirely different
  hardware and backend (AMD ROCm / Windows vs. the original Apple Silicon run): **86.0 vs 88.0
  Top-1, 82.33 vs 84.67 macro-F1.** The result reproduces within ~2 points, which we take as
  the honest backend/seed variance band on a 50-file test set.
- Standing checks still pass: duplicate-content audit (SHA-1 over raw rows) across folds and
  test, train-only standardization, eval-only replay, independent metric recomputation, and a
  shuffled-label control that collapses to 2.0 file Top-1 (chance = 2.0).

## 7. Suggested framing

Primary contribution: **a benchmark-faithful enhanced detector plus the selection-hygiene
protocol** (grouped CV, OOF-only search, single terminal test evaluation) — a significant
+12.0-point file-level gain on the published channel set. Secondary: the channel-quality audit,
the cross-release and cross-platform reproducibility verification, and a 12-channel extension
reported with explicit non-significance.

The negative results are, I think, part of the value here: the extra channels don't hold up
statistically, and saying so makes the headline claim more credible rather than less.

## 8. Key tables

- `results/tables/m05b_final_test.json` — primary 6-channel result
- `results/tables/m05b_vs_exact_upstream_baseline_comparison.json` — headline paired test
- `results/tables/m05b_vs_m05_channel_comparison.json` — 6ch vs 12ch (non-significant)
- `results/tables/m05b_vs_m05_rocm_samemachine_comparison.json` — same-machine 6ch vs 12ch
- `results/tables/m05b_cv_model_bank.csv` — bank ranking and locked aggregators
- `results/tables/channel_quality_audit.csv` — per-channel data quality
- `results/tables/rocm_repro/m05_final_test.json` — cross-platform reproduction
