$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

if (-not (Test-Path "results\pilot_protocol\dreamer_pilot5.json")) {
    & "$PSScriptRoot\build_dreamer_pilot5.ps1"
}

$CommonArgs = @(
    "experiments\crossSubject_geosem_stda_sgda.py",
    "--dataset_name", "dreamer",
    "--pilot_mode",
    "--pilot_config", "results\pilot_protocol\dreamer_pilot5.json",
    "--pilot_target_limit", "1",
    "--pilot_output_name", "dreamer_pilot5_smoke",
    "--epochs", "2",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--sample_length", "3",
    "--stride", "1",
    "--seed", "42",
    "--session_ids", "1",
    "--dreamer_labeltype", "val",
    "--dreamer_ea",
    "--reliability_warmup_epochs", "1",
    "--sparse_k_max", "6",
    "--mmd_schedule", "warmup_cosine_decay",
    "--mmd_confidence_gate", "entropy",
    "--lambda_max", "0.05",
    "--lambda_min", "0.0",
    "--resgca_geo_tau", "1.0",
    "--resgca_geo_weight", "1.0",
    "--proto_tau", "0.07",
    "--fusion_tau", "0.5",
    "--topk", "6",
    "--shrinkage", "0.1",
    "--spd_eps", "1e-5",
    "--geometry_batch_size", "128",
    "--st_dim", "128",
    "--graph_dim", "64",
    "--graph_heads", "4",
    "--adapter_bottleneck", "32",
    "--heads", "4",
    "--dropout", "0.3",
    "--weight_decay", "1e-4",
    "--eval_interval", "1",
    "--log_interval", "1"
)

$Variants = @(
    "proto_only",
    "conditional",
    "resgca_topk",
    "resgca_all"
)

foreach ($Variant in $Variants) {
    Write-Host "===== Smoke DREAMER Pilot-5: $Variant ====="
    python @CommonArgs --experiment_variant $Variant
}
