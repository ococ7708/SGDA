$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

python experiments\crossSubject_geosem_stda_sgda.py `
    --dataset_name dreamer `
    --build_pilot `
    --pilot_config results\pilot_protocol\dreamer_pilot5.json `
    --sample_length 3 `
    --stride 1 `
    --dreamer_labeltype val `
    --dreamer_ea `
    --shrinkage 0.1 `
    --spd_eps 1e-5 `
    --geometry_batch_size 128 `
    --seed 42
