# CA2 Notebooks — Quick Refresher

## eda_vae.ipynb (was eda.ipynb)
1. Load data:
   - 1.1 load CIFAR10
   - 1.2 inspect shapes/dtypes and class names
2. Class balance — bar chart (confirmed even, 6000/class train).
3. Visualize sample images — one row per class.
4. Pixel value histogram (raw 0-255) → justifies normalizing later.
5. Per-channel (R/G/B) mean/std, overall and per class.
6. Per-class pixel variance → proxy for how "visually messy" a class is.
7. Mean image per class (average all images in a class into one).
8. PCA/t-SNE scatter → which classes visually overlap (e.g. cat/dog).
9. Color vs. grayscale preview → for the B&W discussion question.
10. EDA summary — key takeaways.

No cleaning needed — CIFAR10 is already a clean benchmark dataset.

## vae_baseline.ipynb (was vae.ipynb) — trains the baseline CVAE only, full rigor
1. Imports/setup.
2. Approach note — why conditional VAE fits this task.
3. Preprocessing:
   - 3.1 normalize: cast to float32, scale to 0-1
   - 3.2 encode labels: one-hot the 10 labels (for conditioning)
   - 3.3 train/val split: shuffle, carve validation set from the 50k train
   - (augmentation is NOT here anymore — moved to vae_improvement.ipynb Experiment 6, since it's a tunable that belongs with the other experiments, not a baseline default)
4. Sampling layer (reparameterization trick):
   - 4.1 motivation (+ diagram), 4.2 objective (+ diagram), 4.3 implementation, 4.4 alternatives
5. Conditional encoder — 5.1 what it does + diagram, 5.2 implementation.
6. Conditional decoder — 6.1 what it does + diagram, 6.2 implementation.
7. CVAE training step:
   - 7.1 loss formulation, 7.2 architecture pseudocode (full train_step block, exempt from one-liner rule), 7.3 implementation
8. Train the model:
   - 8.1 fit and save best weights, 8.2 loss curves
   - 8.3 reconstruction sanity check (encode/decode real val images, MSE)
   - 8.4 per-class reconstruction error (bar chart)
9. Latent space visualization:
   - 9.1 PCA scatter by class, 9.2 latent interpolation, 9.3 per-dimension effect grid
   - 9.4 aggregate posterior check (z_mean/z_var vs N(0,1)), 9.5 class centroid distance heatmap
10. Generate 1000 class-conditioned images:
    - 10.1 sample and decode (fresh z from prior + label, decoder only — NOT reconstruction)
    - 10.2 save to disk (organized by class folder), 10.3 preview grid (inline sanity check)
11. Evaluate generated image quality:
    - 11.1 eye-test scoring (clear/marginal/nonsense; FID/IS noted as optional, not implemented)
    - 11.2 per-class score summary (bar chart)
    - 11.3 discussion: class difficulty (evidence-based, from this notebook's own eye-test)
    - 11.4 discussion: color vs B&W — PREDICTION ONLY (reasoning from EDA, no grayscale model trained here); explicitly hands off to vae_improvement.ipynb Section 7 for the empirical answer, so this isn't duplicated
12. Baseline conclusion — saves config/losses/eye-test scores to JSON for vae_improvement.ipynb to load (no retraining needed there).

## vae_improvement.ipynb (new, renamed from planned improvements_vae.ipynb) — lighter touch than baseline
Each experiment only does: build+train, then generate → score → compare vs baseline.
No repeat of baseline's heavier diagnostics (recon sanity check, per-class error, posterior check, centroid heatmap) — those were for justifying the baseline's architecture once, not for every variant.
1. Imports/setup (encoder/decoder-building code re-defined here directly, not shared via a .py file — keeps the notebook/.html export self-contained for grading).
2. Load baseline results — same preprocessing, reads baseline's saved JSON, no retraining.
3. Comparison metrics — defined up front, before any experiment runs, so results can't be judged after the fact:
   - 3.1 quality score (primary): eye-test labels (clear=1/marginal=0.5/nonsense=0) → score_from_labels() → 0-1 average, per class and overall
   - 3.2 validation loss (secondary/diagnostic): final val total + val reconstruction loss, catches overfitting/collapse the quality score might miss
   - 3.3 decision rule: quality score ranks; val loss is only a tiebreaker/red-flag check
   - shared results dict initialized here (seeded with baseline), every experiment appends its own quality score + val loss to it
   - NOTE: experiments 1-4 log metrics only, no .h5 weights saved for them — only baseline and the Section 8 final model save weights
4. Experiment 1: latent_dim (bigger vs baseline) — hypothesis: more capacity to represent CIFAR10's variety, at the risk of a less regularized space.
5. Experiment 2: kl_weight / beta-VAE (lower vs baseline) — hypothesis: sharper reconstructions, at the risk of a less structured latent space.
6. Experiment 3: architecture depth (deeper conv stack vs baseline) — hypothesis: more capacity via depth to capture finer textures, at the risk of harder optimization.
7. Experiment 4: color vs grayscale — hypothesis tied directly to the assignment's B&W discussion question; discussion cell here explicitly confirms/contradicts the prediction made in vae_baseline.ipynb 11.4, citing per-class score evidence (this is the ONE place the actual answer lives — not duplicated in baseline).
8. Experiment 5: engineered conditioning features — hypothesis: concatenating each class's mean per-channel color stats (from vae_eda.ipynb Section 5) onto the one-hot label gives richer conditioning than one-hot alone. Lives in improvement, not preprocessing, per the same rule as augmentation/depth/etc — anything tunable is an experiment, baseline stays plain. Answers rubric's "feature engineering (if desirable)" line with an actual tested addition, not just an EDA-stage justification note.
9. Experiment 6: data augmentation — MOVED here from vae_baseline.ipynb's old 3.4 (was a with/without comparison sitting inside the "plain baseline," inconsistent with the baseline-stays-plain rule). Hypothesis: random flip exposes more per-class variation, should reduce overfitting to the exact 5000 training images per class.
10. Final model — apply the 3.3 decision rule to combine winning settings from experiments 1-3, 5, and 6 into the single final model (Experiment 4/color-grayscale is evaluated separately, not folded in — it targets a discussion question, not a quality lever). Only model besides baseline whose weights are saved:
   - 10.1 select best settings (table, from the shared results dict)
   - 10.2 train the final model (training cell), then save its weights to .h5 in a separate dedicated cell
   - 10.3 evaluate vs baseline
11. Ablation summary — ONE table + ONE quality-score bar chart, both built from the shared results dict (no new comparison step, just surfaces numbers already logged during experiments 1-6 and the final model). Trimmed down from an earlier heavier version (4 separate plots) — the numbers already back up the final model choice, a light summary is enough.
12. Conclusion — notes the final model's .h5 is the Part A weights deliverable, alongside the baseline's, and that results feed vae_gan_comparison.ipynb.

## gan_baseline.ipynb (new) — trains the baseline conditional DCGAN only, full rigor
Reuses eda_vae.ipynb — no separate GAN EDA, same CIFAR10 dataset/findings apply.
Baseline is deliberately plain (per user: baseline = comparison point, all tuning happens in gan_improvement.ipynb): no label smoothing, no TTUR, no spectral norm here — hard 0/1 BCE targets, one shared learning rate, single Adam per network.
1. Imports/setup.
2. Approach note — why class-conditional DCGAN (unconditional GAN on all 10 classes at once is notably harder to converge than single-class, so conditioning is baseline not optional).
3. Preprocessing:
   - 3.1 normalize to [-1, 1] (NOT 0-1 — generator uses tanh output, unlike VAE's sigmoid)
   - 3.2 encode labels: one-hot the 10 labels (for conditioning), 3.3 train/val split
   - (augmentation is NOT here anymore — moved to gan_improvement.ipynb Experiment 6, same reasoning as the VAE side)
4. Conditional generator — 4.1 what it does + diagram, 4.2 implementation (noise+label -> transposed convs -> tanh).
5. Conditional discriminator — 5.1 what it does + diagram, 5.2 implementation (image+label map -> strided convs -> logit).
6. GAN training step:
   - 6.1 loss formulation (adversarial BCE, hard targets), 6.2 pseudocode (alternating D-then-G update block), 6.3 implementation
7. Train the model:
   - 7.1 fit + save weights at fixed final epoch (GANs have no single val loss to checkpoint against, unlike VAE)
   - 7.2 loss curves — G and D loss on the SAME axes (their balance matters, not either curve alone)
8. Diagnostics (baseline-only, heavier than experiments get):
   - 8.1 fixed-noise progression grid (same z+label decoded every N epochs — "watch it learn" panel)
   - 8.2 mode-collapse quantification (pairwise pixel-distance/std-dev across same-class batch — numeric, not eyeballed)
   - 8.3 nearest-neighbor check vs real training images (memorization sanity check)
9. Generate 1000 class-conditioned images — 9.1 sample+generate (fresh noise, generator only), 9.2 save to disk, 9.3 preview grid.
10. Evaluate generated image quality:
    - 10.1 eye-test scoring (SAME criteria as CVAE baseline, for direct comparability), 10.2 per-class score summary
    - 10.3 discussion: class difficulty (evidence-based, compares pattern to CVAE's)
    - 10.4 discussion: color vs B&W — PREDICTION ONLY, hands off to gan_improvement.ipynb Section 7 for the empirical answer
11. Baseline conclusion — saves config/losses/eye-test scores to JSON for gan_improvement.ipynb to load.

## gan_improvement.ipynb (new) — lighter touch than baseline, mirrors vae_improvement.ipynb's shape
Researched against real-world CIFAR10 GAN practice (DCGAN/cGAN tutorials, label smoothing, spectral norm, TTUR literature) before finalizing — see conversation for sources.
1. Imports/setup (generator/discriminator/train_step re-defined here directly, not shared via .py — self-contained for grading).
2. Load baseline results — same preprocessing, reads baseline's saved JSON, no retraining.
3. Comparison metrics — defined up front:
   - 3.1 quality score (primary): SAME score_from_labels() definition as vae_improvement.ipynb, so GAN and VAE results are directly comparable
   - 3.2 G/D loss balance (secondary/diagnostic): final gap between generator and discriminator loss — the GAN analogue of VAE's validation loss, since GANs have no single val loss
   - 3.3 decision rule: quality score ranks; loss balance is only a tiebreaker/red-flag check
   - shared results dict initialized here (seeded with baseline); NOTE: experiments 1-4 log metrics only, no .h5 saved — only baseline and Section 8 final model save weights
4. Experiment 1: TTUR (unequal G/D learning rates) — hypothesis: prevents discriminator from outrunning generator.
5. Experiment 2: label smoothing (real target 0.9 vs 1.0) — hypothesis: keeps discriminator gradients informative, less overconfident.
6. Experiment 3: spectral normalization on discriminator — hypothesis: caps discriminator's Lipschitz constant, stabilizes training (swapped in over plain "architecture depth" since vae_improvement.ipynb already covers depth as an axis — avoids redundancy between the two notebooks).
7. Experiment 4: color vs grayscale — hypothesis tied to assignment's B&W question; discussion cell explicitly confirms/contradicts the prediction made in gan_baseline.ipynb 10.4 (this is the ONE place the GAN's actual answer lives).
8. Experiment 5: engineered conditioning features — same idea as vae_improvement.ipynb's Experiment 5 (per-class color stats concatenated onto one-hot), reusing the same shared EDA source so both architectures' feature-engineering experiment draws from identical data.
9. Experiment 6: data augmentation — MOVED here from gan_baseline.ipynb's old 3.4, same reasoning as the VAE side (baseline stays plain, augmentation is a tunable).
10. Final model — apply the 3.3 decision rule to combine winning settings from experiments 1-3, 5, and 6 into the single final model (Experiment 4/color-grayscale evaluated separately, not folded in). Only model besides baseline whose weights are saved:
   - 10.1 select best settings, 10.2 train + save weights in a dedicated cell, 10.3 evaluate vs baseline
11. Ablation summary — ONE table + ONE quality-score bar chart, built from the results dict (same lightweight approach as vae_improvement.ipynb).
12. Conclusion — notes final model's .h5 is the deliverable, feeding into vae_gan_comparison.ipynb.

## vae_gan_comparison.ipynb (new) — the cross-model deliverable, does no training
Confirmed by user as a required deliverable, not optional — loads BOTH final models (vae_improvement.ipynb's and gan_improvement.ipynb's saved .h5 + results JSON) and decides, with evidence, which architecture wins. This is what closes the loop on building two full model families for the same task.
1. Imports/setup (minimal decoder/generator-loading code re-defined here, self-contained for grading).
2. Load final models and results — no training here at all, purely reads what the two improvement notebooks already produced.
3. Quantitative comparison:
   - 3.1 overall + per-class quality score table and grouped bar chart (valid comparison because both notebooks used the IDENTICAL score_from_labels() definition throughout)
   - 3.2 training stability — VAE's val loss curve and GAN's G/D loss curves shown side by side, each on its own natural terms (not forced into one shared number)
4. Qualitative comparison:
   - 4.1 side-by-side sample grid, same classes from both models in matched rows
   - 4.2 diversity check — same pixel-diversity metric from gan_baseline.ipynb 8.2, computed for both models' same-class batches
5. Discussion: which architecture wins and why — verdict tied to mechanism (VAE reconstruction-loss blur vs. GAN adversarial sharpness/instability tradeoff), not just "number X is bigger."
6. Conclusion — headline finding; notes both final models' weights remain actual deliverables regardless of the verdict, since the assignment allows submitting either or both.

## Rubric gaps addressed this session (via brutal-lecturer-critic review)
- Presentation/Demo (9 marks): still untouched — deliberately deferred, not forgotten. Revisit once notebooks have real content to draw slides from.
- Cross-model comparison: was a missing deliverable, now vae_gan_comparison.ipynb above.
- Feature engineering (rubric line): was implicit/absent, now Experiment 5 in both improvement notebooks — an actual tested addition, not just a justification sentence.
- Deliverables checklist: flagged as a gap, explicitly deferred by user for a later session — not tracked yet.

Optional/stretch, not committed to any notebook's core plan: FID score via pretrained InceptionV3 (assignment lists it as optional) — add only if time permits after core notebooks are solid.

Still just skeletons — headings + one-line placeholders, no real code written yet in any of the five notebooks.
