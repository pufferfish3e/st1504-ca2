# CA2 Notebooks — Quick Refresher
`vae_gan_eda.ipynb` and `vae_baseline.ipynb` are fully implemented and verified (see updated section below). The other 4 notebooks are still skeletons (headings + one-liners), no real code written yet.

## vae_gan_eda.ipynb — shared EDA, no cleaning needed (clean benchmark dataset)
1. Load data (1.1 load, 1.2 inspect shapes/classes)
2. Class balance — bar chart (even, 6000/class)
3. Sample images, one row/class
4. Pixel histogram (0-255) → justifies normalization
5. Per-channel R/G/B mean/std, overall + per class
6. Per-class pixel variance → visual-complexity proxy
7. Mean image per class
8. Class similarity heatmap — pairwise dist between mean images, cross-checks Section 9
9. PCA/t-SNE scatter — class overlap (cat/dog etc.), cross-refs Section 8
10. Color vs. grayscale preview → B&W discussion question
11. Summary

## vae_baseline.ipynb — baseline CVAE, full rigor — DONE (verified end-to-end, 2026-07-14)
FID was dropped entirely (see FID note below, now outdated/reversed) in favor of eye-test + CNN reclassifier + CLIP zero-shot as the three quality metrics. All sections implemented and rerun cleanly (30 epochs, train/val loss converge to ~1817.7/1820.5, no overfitting).
1. Imports
2. Approach note
3. Preprocessing: 3.1 normalize 0-1, 3.2 one-hot labels, 3.3 train/val split (augmentation NOT here — moved to improvement Exp 6)
4. Sampling layer (reparam trick): 4.1 motivation, 4.2 objective/impl (Sampling module), 4.3 alternatives
5. Conditional encoder (5.1 diagram, 5.2 impl)
6. Conditional decoder (6.1 diagram, 6.2 impl)
7. Training step: 7.1 loss formulation, 7.2 pseudocode, 7.3 impl (CVAE class)
8. Train: 8.1 fit+save weights (DataLoader-based), 8.2 loss curves, 8.3 recon sanity check (val MSE ~0.011), 8.4 per-class recon error (automobile/truck worst, deer/ship/bird best)
9. Latent space viz: 9.1 PCA scatter, 9.2 interpolation, 9.3 per-dim effect grid (only dim 3 shows a visible effect), 9.4 posterior check (bimodal z_var → partial posterior collapse, ~half the latent dims unused), 9.5 class centroid heatmap
10. Generate 1000 images: 10.1 sample+decode, 10.2 preview grid, 10.3 save to disk (swapped order + renamed from original skeleton)
11. Evaluate quality (FID replaced): 11.1 eye-test (manual c/m/n scoring, primary metric), 11.2 per-class tally, 11.3 discussion class difficulty (cat best, automobile/truck worst), 11.4 discussion color/B&W (PREDICTION only → answered in vae_improvement Exp 4), 11.5 CNN classifier reclassification (73% test acc; agrees with eye-test on cat/automobile/truck, disagrees on frog/dog), 11.6 CLIP zero-shot (collapsed onto predicting "dog" for nearly everything — treated as a failed/uninformative metric, not a ranking), 11.7 eye-test/CNN/CLIP comparison
12. Conclusion — saves config/losses/eye-test/CNN/CLIP results to `vae_baseline_results.json` for vae_improvement.ipynb (no FID field anymore)

## vae_improvement.ipynb — lighter touch, experiments only log metrics (no .h5) except final model
Builders re-defined here (self-contained, no shared .py). No repeat of baseline's heavy diagnostics.
1. Imports
2. Load baseline results (same preprocessing, reads JSON, no retrain)
3. Comparison metrics, defined up front: 3.1 quality score (primary, score_from_labels), 3.2 val loss (secondary/diagnostic), 3.3 decision rule (quality score ranks, val loss = tiebreaker), shared results dict; FID tracked alongside as non-ranking quantitative check (loaded from baseline, recomputed for final model only, ablation table column)
4. Exp 1: latent_dim ↑ — more capacity vs. less regularized space
5. Exp 2: kl_weight ↓ (beta-VAE) — sharper recon vs. less structured latent
6. Exp 3: architecture depth ↑ — finer textures vs. harder optimization
7. Exp 4: color vs grayscale — answers assignment's B&W question empirically (vs. baseline's 11.4 prediction)
8. Exp 5: engineered conditioning features — per-class RGB stats concat onto one-hot; answers rubric's feature-engineering line
9. Exp 6: data augmentation (moved from baseline 3.4) — random flip vs. overfit risk
10. Final model: 10.1 select best settings (combines winners of 1-3,5,6; Exp4 evaluated separately), 10.2 train+save .h5, 10.3 evaluate vs baseline (+ FID)
11. Ablation summary — one table + one bar chart from results dict
12. Conclusion — .h5 is Part A deliverable, feeds vae_gan_comparison.ipynb

## gan_baseline.ipynb — baseline conditional DCGAN, full rigor
Reuses shared EDA. Deliberately plain: no label smoothing/TTUR/spectral norm (those are improvement experiments).
1. Imports
2. Approach note — why conditional (unconditional GAN across 10 classes converges worse)
3. Preprocessing: 3.1 normalize [-1,1] (tanh output), 3.2 one-hot labels, 3.3 train/val split (augmentation moved to improvement Exp 6)
4. Conditional generator (4.1 diagram, 4.2 impl: noise+label → transposed convs → tanh)
5. Conditional discriminator (5.1 diagram, 5.2 impl: image+label map → strided convs → logit)
6. Training step: 6.1 loss formulation (hard BCE targets), 6.2 pseudocode, 6.3 impl
7. Train: 7.1 fit+save at fixed epoch (no single val loss to checkpoint against), 7.2 G/D loss curves same axes
8. Diagnostics (baseline-only): 8.1 fixed-noise progression grid, 8.2 mode-collapse quantification (pairwise pixel dist), 8.3 nearest-neighbor memorization check
9. Generate 1000 images: 9.1 sample+generate, 9.2 save to disk, 9.3 preview grid
10. Evaluate quality: 10.1 eye-test (same criteria as CVAE), 10.2 per-class summary, 10.3 discussion class difficulty, 10.4 discussion color/B&W (PREDICTION → answered in gan_improvement Exp 4), 10.5 FID score (same InceptionV3 extractor as VAE side)
11. Conclusion — saves config/losses/eye-test/FID JSON for gan_improvement.ipynb

## gan_improvement.ipynb — mirrors vae_improvement.ipynb's shape
1. Imports (builders re-defined, self-contained)
2. Load baseline results
3. Comparison metrics: 3.1 quality score (SAME score_from_labels as VAE side — comparable), 3.2 G/D loss balance (secondary/diagnostic, GAN's analogue of val loss), 3.3 decision rule; FID tracked alongside, same treatment as VAE side
4. Exp 1: TTUR (unequal G/D learning rates) — prevents D outrunning G
5. Exp 2: label smoothing (0.9 real target) — keeps D gradients informative
6. Exp 3: spectral norm on D — caps Lipschitz constant, stabilizes (chosen over "depth" to avoid redundancy with VAE's Exp 3)
7. Exp 4: color vs grayscale — answers B&W question empirically (vs. baseline's 10.4 prediction)
8. Exp 5: engineered conditioning features — same idea/source as VAE side
9. Exp 6: data augmentation (moved from baseline 3.4)
10. Final model: 10.1 select best (combines 1-3,5,6; Exp4 separate), 10.2 train+save .h5, 10.3 evaluate vs baseline (+ FID)
11. Ablation summary — one table + one bar chart
12. Conclusion — .h5 deliverable, feeds vae_gan_comparison.ipynb

## vae_gan_comparison.ipynb — cross-model deliverable, no training
Required deliverable (confirmed). Loads both final models' .h5 + results JSON.
1. Imports (minimal loading code, self-contained)
2. Load final models + results
3. Quantitative: 3.1 quality score table/chart + FID column (both use identical scoring/extractor — valid comparison), 3.2 training stability side by side (VAE val loss vs GAN G/D balance, own terms)
4. Qualitative: 4.1 side-by-side sample grid, 4.2 diversity check (reuses gan_baseline 8.2's metric)
5. Discussion — verdict tied to mechanism (VAE blur vs GAN sharpness/instability), not just numbers
6. Conclusion — headline finding; both models' weights remain deliverables regardless of verdict

## Rubric gaps addressed (brutal-lecturer-critic review)
- Presentation/Demo (9 marks): deliberately deferred, revisit once notebooks have real content
- Cross-model comparison: was missing → vae_gan_comparison.ipynb
- Feature engineering: was implicit → Experiment 5 in both improvement notebooks
- Deliverables checklist: deferred by user, not tracked yet

## FID note — REVERSED for vae_baseline.ipynb (2026-07-14)
FID was dropped from `vae_baseline.ipynb` entirely (all mentions, code, JSON field) in favor of eye-test + CNN reclassifier + CLIP zero-shot. The plan below (written when FID was promoted to core) is now stale for the VAE side. Still need to decide: drop FID from `gan_baseline.ipynb`/`gan_improvement.ipynb`/`vae_gan_comparison.ipynb` too for consistency, or keep it there since GAN literature leans on FID more heavily than VAE literature does — not yet decided.

## FID note (ORIGINAL — promoted from optional/stretch to core, via CIFAR10 lit check; superseded above for the VAE side)
- Eye-test-only was flagged as the biggest real risk — FID is the standard quantitative metric everywhere in CIFAR10 gen-model literature. Now implemented in all 5 notebooks (see sections above).
- One-hot concat conditioning is the "weakest" standard method (vs. projection discriminator/conditional batch norm) — NOT worth swapping this late; noted as a limitations/future-work line in the comparison notebook instead.
- EMA of generator weights: cheap, low-risk optional 7th GAN experiment, not yet added.
- Confirmed fine as-is: latent_dim/KL-weight axes, TTUR, label smoothing, spectral norm — standard "cheap wins" tier. Explicitly ruled out: self-attention, progressive growing, hinge/WGAN-GP loss, ResNet blocks — disproportionate for 32x32/10-class coursework.
