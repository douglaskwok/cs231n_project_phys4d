Tier 1 — high impact, data already on disk (quick)
Baseline table + plot on the main 50k run. Re-run baseline_compare.py on step4a_fixedset/trajectory_smoothed.csv + step4c_metric/trajectory_predicted_best.csv + GT poses. Currently this exists only for debug runs — the main result has no baseline contrast. → metrics.json, comparison.csv, compare_plot.png.

Generalization table across variants. You have e0p90, e0p93, e0p78, two radii, and a collision scene staged. A table of fitted-e vs GT-e and held-out RMSE per variant directly kills the "single scene" limitation and is the strongest robustness story.

Smoothing / centroid-estimator ablation table. Consolidate the existing step4b_metric_{raw,w9,w13,w9_robust} runs into one table (window → fitted e, fit MSE, held-out RMSE). Shows the global-fit-vs-noisy-boundary robustness point quantitatively.

Physics-parameter recovery table (p₀, v₀, e, z_g: fitted vs GT, with % error) — emphasizes interpretable recovered params, not learned weights.

Per-camera render metrics (PSNR/SSIM for each of the 12 cameras as a bar chart) — turns the single render number into a distribution.

Tier 2 — qualitative figures the rubric explicitly rewards
Pipeline overview figure (the 5-stage diagram, status-colored) — needed for Intro/Methods.

Dataset example montage — multi-view RGB + SAM2 mask + object-only crop, with split/resolution annotated (Section 4 requires example data).

Failure-case panel + difference heatmap. The render is the honest weak point (ball too small/under-rendered vs GT). The rubric explicitly asks for failure cases and heatmaps: show composite vs GT vs per-pixel error/SSIM map and discuss the cause (background-3DGS holes, ball scale/opacity).

3D trajectory plot (currently only per-axis x/y/z) showing the bounce arc in 3D, ours vs GT vs baselines fanning away.

"No double-motion" ablation figure — render with vs without the deformation field applied in Step 5, demonstrating the freeze-canonical claim (you already have mirror_compare-style assets to build from).

Tier 3 — nice-to-have / framing
4DGS reconstruction-quality table (train-window PSNR/SSIM at 8k vs 50k iters) to justify the longer training and explain the x/y wobble.
Bounce-detection visualization (v_z sign-flips over the train window).
A short note in the report that classification-style analyses (t-SNE, confusion matrices, saliency) are not applicable to this regression/rendering task — the held-out time split is the overfitting control, so report train-fit MSE vs test RMSE as the overfitting analysis.
Tiers 1–2 are mostly assembly of data you already have, so they're the highest value-per-effort. Want me to start generating any of these? I'd suggest beginning with #1 (main-run baselines) and #2 (generalization table) since they're the load-bearing quantitative claims, then the #8 failure/heatmap figure.

Here's a paste-ready metrics summary for the Experiments/Results section, organized to hit each written requirement in Section 5 of the rubric (primary metrics + equations, hyperparameters, quantitative tables, and overfitting). All numbers are pulled from the artifacts in outputs/bounce_pipeline/wu_ball12_2s_blue_e93_iter50k/ and the two debug-scene baseline runs.

5.1 Primary metrics (definitions + equations)
Position RMSE (held-out trajectory, the headline metric):

RMSE
=
1
T
∑
t
∥
p
^
(
t
)
−
p
g
t
(
t
)
∥
2
RMSE=T1​t∑​∥p^​(t)−pgt​(t)∥2​
Velocity / acceleration R² — Pearson 
R
2
R2 between predicted and GT finite-difference derivatives 
p
˙
,
p
¨
p˙​,p¨​ over held-out frames. Reported because RMSE alone doesn't capture whether the dynamics (bounce timing, gravity) are right.

Physics-parameter recovery error — relative error of the fitted closed-form parameters vs PyBullet GT, e.g. 
∣
e
f
i
t
−
e
g
t
∣
/
e
g
t
∣efit​−egt​∣/egt​. This is the interpretability claim: parameters are recovered, not learned weights.

PSNR on held-out composite renders: 
PSNR
=
10
log
⁡
10
 ⁣
(
1
/
MSE
)
PSNR=10log10​(1/MSE) (images in 
[
0
,
1
]
[0,1]).

SSIM — structural similarity (luminance·contrast·structure), and MAE as an auxiliary appearance error.

5.2 Hyperparameters and how they were chosen
Component	Setting	How chosen
Wu 4DGS training
50k iters (resumed from 30k), SH degree 3
longer than 8k/30k debug runs to cut x/y centroid jitter
Train/test split
train frames 0–240 (241), held-out 241–360 (120) ≈ 2/3 : 1/3
held-out time split; recorded in frame_map.json
Trajectory smoothing
Savitzky–Golay, window = 9, polyorder = 2
ablation (§5.5): w9 ≈ w13, both beat raw and robust
Physics fit
scipy.optimize.least_squares, soft_l1 robust loss, f_scale=0.05, gravity fixed −9.81 m/s²
LM/soft_l1 for jitter robustness; g fixed for first run, free as ablation
Free parameters
11 scalars: p₀(3), v₀(3), e, z_g (+g)
closed-form, no network
No k-fold CV: validation is the held-out time split, and generalization is checked across scene variants (restitution e ∈ {0.78, 0.90, 0.93}, two radii) rather than folds.

5.3 Main result — held-out trajectory (50k run, e=0.93 scene)
Metric	Result	Target	Pass
Position RMSE
4.06 cm
< 5 cm
✅
Per-frame error
mean 3.4 cm, max 8.5 cm
—
—
Fitted restitution e
0.901 vs GT 0.93 (−3.1%)
—
✅
Train-window fit MSE
1.5×10⁻³ m²
< 0.01 m²
✅
Bounces detected
3 (matches plot)
—
✅
Velocity R²
0.27
> 0.9
❌
Honest caveat: velocity R² is low on this run because residual x/y centroid wobble dominates the finite-difference derivative; the position and bounce-timing fit is strong (z-axis curve in analysis_xyz_best.png).

5.4 Baseline comparison (held-out RMSE, m)
Method	Big ball r=0.35	Small ball r=0.20
4DGS-only (freeze last centroid)
0.0359
0.0746
Constant velocity
0.0859
0.2080
Ballistic (gravity, no bounce)
0.0185
0.1476
Ours (physics fit)
0.0086
0.0382
Robustness point: on the small ball the naive boundary extrapolators blow up (const-vel 20.8 cm) because a single noisy boundary velocity corrupts them; our global fit over the whole train window stays at 3.8 cm. (Baselines currently exist only for the two debug scenes — Tier-1 item #1 is to run this on the 50k scene.)

5.5 Smoothing ablation (50k run)
Smoothing	Held-out RMSE	Train fit MSE	Fitted e
raw (no smoothing)
12.1 cm
7.1×10⁻³
0.915
Sav-Gol w9
4.06 cm
1.5×10⁻³
0.901
Sav-Gol w13
4.06 cm
1.48×10⁻³
0.901
w9 + robust loss
5.57 cm
2.4×10⁻³
0.903
5.6 Rendering metrics (held-out composite, 240 frame×camera pairs)
Metric	Result	Target	Pass
PSNR
15.8 dB
> 25 dB
❌
SSIM
0.88
> 0.85
✅
MAE
0.06
—
—
The trajectory (what we predict) passes; PSNR is the honest weak point — the composited ball is under-rendered vs GT (render_compare_best.png), a property of the appearance pipeline, not the prediction.

5.7 Overfitting analysis
The physics model has only 11 free scalars fit to 241 observed frames — massively overdetermined, so NN-style overfitting is impossible by construction; the closed-form bounce ODE is the regularizer. Evidence: train-window fit MSE (1.5×10⁻³ m²) and held-out RMSE² (≈1.6×10⁻³ m²) are nearly equal, i.e. essentially no train→test generalization gap. Classification-style diagnostics (t-SNE, confusion matrices, saliency maps) are not applicable to this regression+rendering task; the held-out time split is the overfitting control.