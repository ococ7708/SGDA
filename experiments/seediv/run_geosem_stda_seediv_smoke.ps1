$ErrorActionPreference = "Stop"

$Python = "C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe"
$Script = Join-Path $PSScriptRoot "crossSubject_geosem_stda_seediv.py"

& $Python -u $Script `
  --target_subject_ids 1 `
  --source_selection sparse_reliability `
  --epochs 2 `
  --batch_size 32 `
  --lr 1e-3 `
  --reliability_warmup_epochs 1 `
  --sparse_k_max 3 `
  --eval_interval 1 `
  --log_interval 1 `
  --lambda_max 0.2 `
  --lambda_min 0.05 `
  --mmd_type resgca `
  --mmd_schedule warmup_cosine_decay `
  --mmd_confidence_gate entropy `
  --resgca_geo_tau 1.0 `
  --resgca_geo_weight 1.0 `
  --topk 8 `
  --geometry_batch_size 64 `
  --st_dim 128 `
  --graph_dim 64 `
  --adapter_bottleneck 32 `
  --heads 4 `
  --dropout 0.3 `
  --weight_decay 1e-4
