# DC-CPETNet

This repository contains the source code used to implement DC-CPETNet, including data preprocessing, model training, evaluation, ablation, and interpretability analysis scripts, together with trained model weights.

## Contents

| Component | Main files |
|---|---|
| Data preprocessing | `src/data_preprocess_new.py`, `src/dataset_new.py`, `src/dataset_mtl.py`, `src/label_extractor.py`, `src/feature_mapping.py` |
| Model training | `src/main_mtl.py`, `src/train_mtl_with_swanlab.py`, `src/main_new.py`, `src/train_with_swanlab.py` |
| Model architecture | `src/model_mtl.py`, `src/model.py`, `src/models/`, `src/modules/`, `src/baselines/` |
| Evaluation | `src/main_mtl.py --resume --stage holdout`|
| Ablation | `configs/config_mtl_ablation_*.yaml` |
| Comparison baselines | `configs/clean/*.yaml`, `src/models/architectures/*_clean_v4.py`, `src/training/plans/*_clean_plan.py` |
| Interpretability analysis | `src/main_mtl.py --interpret_enabled`, `tools/summarize_gate_usage.py`, `scripts/cpet_interpret_plot_scripts/` |




## Environment

Use Python 3.10 with PyTorch and the packages in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

On Windows, the following environment variables help avoid encoding and buffering issues:

```powershell
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
```

## Holdout Evaluation

The fixed MTL holdout split file is:

```text
results/holdout_split_info_mtl.json
```

Run a single fold holdout test with a trained checkpoint:

```powershell
python -u src/main_mtl.py `
  --mode t6_auxiliary `
  --resume --stage holdout `
  --holdout_enabled `
  --fold 0 `
  --config configs/config_mtl_ablation_B_feat02.yaml `
  --checkpoint models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold1.pth `
  --disable_swanlab
```

For 5-fold holdout evaluation, repeat the command with `--fold 0..4` and the matching `fold1..fold5` checkpoint.

Typical outputs are written to `results/`, `threshold_search_results/`, and `metrics_logs/`, including Excel summaries, ROC data, thresholds, and anonymized holdout prediction tables.

## Trained Weights

The trained DC-CPETNet 5-fold weights are included in `models/`:

```text
models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold1.pth
models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold2.pth
models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold3.pth
models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold4.pth
models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold5.pth
```

These weights can be evaluated directly on the configured holdout set after the authorized data files are available locally.

## Training

Train the main DC-CPETNet model across all folds:

```powershell
python -u src/main_mtl.py `
  --mode t6_auxiliary `
  --holdout_enabled `
  --run_all_folds `
  --config configs/config_mtl_ablation_B_feat02.yaml `
  --disable_swanlab
```

Train one fold:

```powershell
python -u src/main_mtl.py `
  --mode t6_auxiliary `
  --holdout_enabled `
  --fold 0 `
  --config configs/config_mtl_ablation_B_feat02.yaml `
  --disable_swanlab
```

The single-task baseline entry is:

```powershell
python -u src/main_new.py
```

`src/main_new.py` is configuration-driven; change the target task in `configs/config.yaml` before running.



## Ablation Experiments

Main ablation configuration files:

| Experiment | Configuration |
|---|---|
| Our Method | `configs/config_mtl_ablation_B_feat02.yaml` |
| E1 | `configs/config_mtl_baseline.yaml` |
| E2 | `configs/config_mtl_ablation_E2_single_shared_alpha.yaml` |
| E3 | `configs/config_mtl_ablation_E3_beta_shared_only.yaml` |
| E4 | `configs/config_mtl_ablation_E4_alpha_no_prior.yaml` |
| S1 | `configs/config_mtl_ablation_S1_no_t3_private.yaml` |
| S2 | `configs/config_mtl_ablation_S2_no_group245.yaml` |
| S3 | `configs/config_mtl_ablation_S3_no_beta_gates.yaml` |
| S4 | `configs/config_mtl_ablation_S4_alpha_no_pmgt.yaml` |
| S5 | `configs/config_mtl_ablation_S5_alpha_no_multiscale_residual.yaml` |
| S6 | `configs/config_mtl_ablation_S6_alpha_no_prior_teacher_no_prior.yaml` |

Example:

```powershell
python -u src/main_mtl.py `
  --mode t6_auxiliary `
  --holdout_enabled `
  --run_all_folds `
  --config configs/config_mtl_ablation_E2_single_shared_alpha.yaml `
  --disable_swanlab
```

## Comparison Baselines

The clean MTL comparison baselines use the same entry point and switch architecture through `--mode`.

| Baseline | Mode | Configuration |
|---|---|---|
| AdaTT | `adatt_clean` | `configs/clean/adatt_clean.yaml` |
| CGC / PLE-style | `cgc_clean` | `configs/clean/cgc_clean.yaml` |
| MMoE | `mmoe_clean` | `configs/clean/mmoe_clean.yaml` |
| Shared-bottom | `shared_bottom_clean` | `configs/clean/shared_bottom_clean.yaml` |

```powershell
python -u src/main_mtl.py --mode adatt_clean --holdout_enabled --run_all_folds --disable_swanlab
python -u src/main_mtl.py --mode cgc_clean --holdout_enabled --run_all_folds --disable_swanlab
python -u src/main_mtl.py --mode mmoe_clean --holdout_enabled --run_all_folds --disable_swanlab
python -u src/main_mtl.py --mode shared_bottom_clean --holdout_enabled --run_all_folds --disable_swanlab
```

## Interpretability

Run holdout evaluation with interpretability enabled:

```powershell
python -u src/main_mtl.py `
  --mode t6_auxiliary `
  --resume --stage holdout `
  --holdout_enabled `
  --fold 0 `
  --config configs/config_mtl_ablation_B_feat02.yaml `
  --checkpoint models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold1.pth `
  --disable_swanlab `
  --interpret_enabled `
  --interpret_out_dir results/interpretation/fold1
```

Optional flags:

```text
--interpret_save_attr
--interpret_save_intermediates
--interpret_context_counterfactual
```

Gate usage analysis:

```powershell
python tools/summarize_gate_usage.py `
  --checkpoint models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold1.pth `
  --output results/gate_usage/fold1
```

## Data Availability

The clinical CPET Excel files and the label/summary workbook are not included in this public release because they contain participant-level clinical information. The original local data directory `F:\data_anonymized\excel` and `final_summary_report.xlsx` are available from the corresponding author upon reasonable request and subject to institutional approval and applicable requirements for protecting participant privacy.

Before running training or evaluation, configure these fields in `configs/config.yaml` and `configs/config_mtl_base.yaml`:

```yaml
data:
  data_root: xx_path
  label_file: xx_path
  output_root: xx_path
```

If PFT/static feature fusion is enabled, also configure:

```yaml
pft_file: xx_path
```
## Notes For Public Release

The source and configuration files use `xx_path` wherever user-local data paths are required. Replace these placeholders only in a private local configuration.

Do not include generated caches or local run artifacts in a public source archive:

```text
__pycache__/
*.pyc
.codex_work/
logs/
run_logs/
metrics_logs/
swanlab_local/
swanlog/
```