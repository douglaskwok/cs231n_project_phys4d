# Generative AI — Coding Agent Usage Statement (Draft)

**Course:** CS231N Spring 2026  
**Project:** Physics-Aware 4D Scene Reconstruction for Rigid-Body Trajectory Prediction  
**Authors:** Simon Casper, Sze Heng Douglas Kwok, Janhavi Purkar  
**Tool:** [Cursor](https://cursor.com) IDE with Agent mode (Claude-based coding agent)  
**Policy reference:** CS231N project honor code

---

## 1. Scope of generative AI use

### Used for (code & tooling)

Cursor Agent was used as a **development assistant** for implementation, debugging, and repository maintenance in the portions of the pipeline attributed to Simon Casper (Steps 4a–6: trajectory extraction, physics fit, prediction, Gaussian compositing, evaluation, and Modal orchestration).

Typical uses:

- Drafting and iterating on Python modules under `src/phys4d/` and `scripts/bounce/` / `scripts/collision/`
- Writing shell workflow scripts and Modal CLI wrappers
- Unit tests under `tests/`
- Debugging CUDA/Modal integration, compositing edge cases, and metric pipelines
- Refactoring, docstrings, and command-reference updates (`STAGES_4_6_COMMANDS.txt`, `context.md`, `README.md`)
- Workspace hygiene (repo audit, `.gitignore` fixes)

### Not used for (report)

Per course policy, **generative AI was not used to author the substantive content** of the final report (`CS231n_Final_Report-9.pdf`). AI was limited to **spell-check, grammar, and LaTeX/table formatting** only, as noted in the report disclaimer.

Teammates (Douglas Kwok, Janhavi Purkar) may have used AI independently for data generation, segmentation, and 4DGS training scripts; those contributions should be documented separately if applicable.

---

## 2. Human responsibility & review

All AI-generated code was **reviewed, tested, and edited by a project author** before use. Group members remain responsible for correctness, attribution, and honor-code compliance.

Review practices:

- Every merged change was run through the step-specific unit tests (`tests/test_step4*.py`, `tests/test_step5*.py`, `tests/test_step6_metrics.py`)
- Physics equations and loss formulations were checked against the report (§4.2–4.5) and cited references (Umeyama, trust-region LS, Weiszfeld median, Savitzky–Golay)
- Hyperparameters and compositing thresholds were tuned manually on held-out scenes
- Modal GPU jobs were executed and inspected by the author (render PNGs, trajectory CSVs, `metrics.json`)

---

## 3. AI-generated or AI-assisted artifacts (by area)

> **Note:** This table is a draft inventory. Mark `[AI-DRAFT]` where the agent produced the first version; `[AI-ASSIST]` where the agent helped refactor/debug existing human code; `[HUMAN]` where written without agent involvement. **Simon: fill in final classifications before submission.**

| Area | Paths | Typical AI role |
|------|-------|-----------------|
| Trajectory extraction (4a) | `src/phys4d/bounce/extract.py`, `load_4dgs.py`, `scripts/bounce/step4a_extract.py`, `scripts/collision/step4a_extract.py` | [AI-DRAFT/ASSIST] kNN density clustering, Weiszfeld centroid, Wu checkpoint loading |
| Physics fit (4b) | `src/phys4d/bounce/physics.py`, `metric_refit.py`, `src/phys4d/collision/physics.py`, `scripts/*/refit_metric_gt.py` | [AI-DRAFT/ASSIST] Umeyama alignment, bounce/collision simulators, scipy least-squares wrappers |
| Prediction (4c) | `src/phys4d/bounce/extrapolate.py`, `scripts/*/predict_metric_from_refit.py` | [AI-DRAFT/ASSIST] Euler forward integration |
| Scene compositing (5) | `src/phys4d/bounce/render_compose.py`, `src/phys4d/collision/render_compose.py`, `gaussian_ply.py`, `scripts/*/step5_render.py` | [AI-DRAFT/ASSIST] depth-guided alpha blend (Eq. 20–21), multi-object layering |
| Evaluation (6) | `src/phys4d/bounce/metrics.py`, `scripts/*/step6_eval.py` | [AI-DRAFT/ASSIST] RMSE, PSNR, SSIM, MAE |
| Modal orchestration | `modal_app.py` | [AI-ASSIST] GPU job staging, volume upload/download |
| Workflow docs | `scripts/bounce/workflow.md`, `scripts/collision/workflow.md` | [AI-ASSIST] reproducibility notes |
| Tests | `tests/test_step4a_timestamps.py` … `test_step6_metrics.py` | [AI-DRAFT] synthetic fixtures, regression checks |
| Dev tooling (not submitted) | `.cursor/hooks/*`, `.cursor/rules/*`, `.cursor/permissions.json` | [AI-DRAFT] local Cursor automation |

### Superseded paths (human-authored first, later replaced)

The metric-refit pipeline (`refit_metric_gt.py` → `predict_metric_from_refit.py`) superseded an earlier direct-fit path (`step4b_fit.py`, `step4c_predict.py`). Both may contain AI-assisted edits; the **report and final results use the metric-refit path**.

---

## 4. Prompts & transcripts

Course policy requires documenting **plans, prompts, and transcripts**. For this project:

| Artifact | Location | Status |
|----------|----------|--------|
| Cursor chat transcripts | `~/.cursor/projects/.../agent-transcripts/*.jsonl` (local machine) | **To export:** select sessions covering Steps 4–6 implementation and attach to submission or appendix |
| Reproduction commands | `README.md`, `STAGES_4_6_COMMANDS.txt`, `context.md`, `scripts/bounce/workflow.md` | Included in repo |

**Submission checklist:**

- [ ] Export relevant Cursor transcripts (PDF or `.jsonl`) for the coding sessions that produced Steps 4–6
- [ ] Annotate transcript excerpts that correspond to major modules (extract, refit, compose, eval)
- [ ] Confirm no transcript content was pasted verbatim into the final report prose

---

## 5. External sources vs AI

Code also incorporates **publicly available implementations and papers**, cited in the report and/or source comments:

- Wu et al. 4DGS (`third_party/4DGaussians` submodule)
- Kerbl et al. 3DGS rasterizer (via Wu dependency)
- PyBullet synthetic data (teammate pipeline)
- Standard libraries: NumPy, SciPy (`least_squares`, Savitzky–Golay), Modal

AI assistance is **in addition to** these sources, not a substitute for citation. Where agent output closely followed a public API or paper equation, the human author verified correctness independently.

---

## 6. Attribution statement (for report appendix)

> We used Cursor Agent (Claude) as a coding assistant for implementing and testing the physics-extraction, prediction, compositing, and evaluation pipeline (Steps 4a–6). All generated code was reviewed, modified, and validated by the authors against synthetic ground truth. Generative AI was not used to write the technical argumentation or results analysis in the final report; it was used only for spell-check and formatting of the written submission. Prompts and agent transcripts for the coding sessions are included in [location TBD].

---

## 7. Per-author attestation (sign before submission)

| Author | AI used for code? | AI used for report prose? | Reviewed all AI code? | Date |
|--------|-------------------|---------------------------|----------------------|------|
| Simon Casper | Yes (Cursor Agent) | No (formatting only) | ☐ | |
| Sze Heng Douglas Kwok | ☐ Yes / ☐ No | ☐ | ☐ | |
| Janhavi Purkar | ☐ Yes / ☐ No | ☐ | ☐ | |
