"""Learned visual dynamics (prediction stage). Implementation TBD on feat/visual-dynamics-reframe.

Planned modules:
  - feature_encoder: CNN / DINO crops per object
  - dynamics_head: Transformer or GNN over (state_history, visual_features)
  - rollout: autoregressive SE(3) prediction + Gaussian warp hook
"""
