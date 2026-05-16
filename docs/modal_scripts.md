# Modal quick reference

Full workflow: **[`WORKFLOW_COMMANDS.md`](WORKFLOW_COMMANDS.md)**

## Visual dynamics (primary)

| Step | Command |
|------|---------|
| Upload batch | `bash scripts/upload_batch_to_modal.sh` |
| Perception | `modal run modal_app.py --extract-perception` |
| Train | `modal run modal_app.py --train-visual-dynamics` |
| Download ckpt | `modal volume get phys4d-gs-output visual_dynamics . --force` |
| 3DGS | `modal run modal_app.py --upload` then `--train` |
| E2E | `modal run modal_app.py --upload-visual-pipeline` then `--visual-pipeline` |

Avoid `modal run modal_app.py --upload-batch` on first run (slow image build); use the upload script.

## Optional 4DGS

`--upload-4d` → `--train-4d` → `--render-4d` → `--eval-4d`

## Tests / smoke

```bash
modal run modal_app.py --tests
modal run modal_app.py                    # GPU smoke only
```
