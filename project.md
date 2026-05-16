Two-stage flips where the NN sits: instead of CNN→params→simulator, the NN is the dynamics and PyBullet is only for generating training data. More ambitious, more end-to-end neural, more CV-flavored — but compounding rollout error becomes the failure mode you have to plan around from day one.
Phase 1 — Data pipeline

 PyBullet generator: video + per-frame per-object states (position, quaternion, velocity) + object masks
 100–200 scenarios, K=10 views, 80/10/10 split by instance
 Decide rollout horizon T_future (start at 30 frames, push longer once stable)
 Save params too — only for ablations, not training signal

Phase 2 — Perception (repurposed 4DGS)

 4DGS reconstruction on training half of each video
 Object→Gaussian clustering with persistent labels across time
 Per-frame state extraction: COM + principal-axis orientation + finite-diff velocity
 Per-object visual features: ResNet/DINO patches pooled over each object's pixels at t=0, fixed per object, not re-extracted later

Phase 3 — Dynamics head architecture (the NN)

 Per-object token: concat(state_history_K, visual_feature)
 Architecture: GNN with edge messages between objects (clean for collisions) or Transformer with object tokens + self-attention
 Output: SE(3) delta per object for t+1
 Rollout function: apply delta → re-feed as next history → iterate T_future times

Phase 4 — Training

 Stage A: 1-step prediction with teacher forcing (cheap, stable warmup)
 Stage B: curriculum to longer rollouts (1 → 5 → 10 → T_future), with noise injection on history
 Losses:

L_state: position MSE + geodesic quaternion loss (not raw quaternion MSE)
L_render: SSIM + L1 on rendered future frames vs GT — this is the CV loss
L_physics (optional): energy non-increase, momentum conservation across contacts


 AdamW, lr=3e-4, gradient clipping (rollout grads are unstable)

Phase 5 — Inference pipeline

 Test video → 4DGS → last K states + visual features per object
 Autoregressive rollout for T_future steps
 Apply predicted SE(3) to per-object Gaussian clusters → render each future frame
 One end-to-end demo working before Milestone 2

Phase 6 — Evaluation & ablations

 Trajectory MSE over horizon (characterize the error growth curve, don't hide it)
 Velocity Pearson R²
 PSNR/SSIM on rendered future frames
 Ablation 1 (justifies CV framing): no visual features → states-only baseline; visual conditioning should win
 Ablation 2: rollout horizon sweep — shows where compounding error dominates
 Ablation 3: cross-scenario generalization (train bounces → test collisions)
 Ablation 4: with/without physics regularization