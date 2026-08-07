$ErrorActionPreference = "Stop"

$Python = "C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe"
$Script = Join-Path $PSScriptRoot "crossSubject_geosem_stda_seediv.py"

& $Python -u $Script `
  --target_subject_ids 1 4 7 10 13 `
  --source_selection sparse_reliability `
  --epochs 200 `
  --batch_size 64 `
  --lr 1e-3 `
  --reliability_warmup_epochs 5 `
  --sparse_k_max 6 `
  --eval_interval 1 `
  --log_interval 1 `
  --lambda_max 0.2 `
  --lambda_min 0.05 `
  --mmd_type resgca `
  --mmd_schedule warmup_cosine_decay `
  --mmd_warmup_ratio 0.2 `
  --mmd_hold_ratio 0.5 `
  --mmd_confidence_gate entropy `
  --mmd_confidence_threshold 0.6 `
  --sca_mu_start 0.0 `
  --sca_mu_end 1.0 `
  --sca_mu_warmup_ratio 0.5 `
  --resgca_geo_tau 1.0 `
  --resgca_geo_weight 1.0 `
  --proto_tau 0.07 `
  --fusion_tau 0.5 `
  --topk 8 `
  --shrinkage 0.1 `
  --spd_eps 1e-5 `
  --geometry_batch_size 128 `
  --st_dim 128 `
  --graph_dim 64 `
  --adapter_bottleneck 32 `
  --heads 4 `
  --dropout 0.3 `
  --weight_decay 1e-4
