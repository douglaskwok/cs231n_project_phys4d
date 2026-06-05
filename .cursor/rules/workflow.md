---
alwaysApply: true
---
always run modal commands s. t. I can see a progress bar
in scripts/bounce (see scripts/bounce/workflow.md):
run step4a_extract.py on modal (default savgol window 9, polyorder 2)
run refit_metric_gt.py
run predict_metric_from_refit.py
run step5_render.py on modal with object scale and object crop radius (estimate a good value), layered alpha compositing (--composite-mode alpha, --object-opacity-boost 4.0), and the fitting background PLY on the volume (bg_ball12blue_med_3dgs/.../30000 for ball scenes)
run step6_eval.py 

create an mp4 for the fused final scene
