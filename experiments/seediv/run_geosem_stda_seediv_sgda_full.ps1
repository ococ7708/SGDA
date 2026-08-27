$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

$RunArgs = @(
    "experiments\crossSubject_geosem_stda_sgda.py",
    "--dataset_name", "seediv",
    "--epochs", "200",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--sample_length", "3",
    "--stride", "1",
    "--seed", "42",
    "--session_ids", "1", "2", "3",
    "--source_selection", "sparse_reliability",
    "--reliability_warmup_epochs", "5",
    "--sparse_k_max", "6",
    "--mmd_type", "resgca",
    "--mmd_schedule", "warmup_cosine_decay",
    "--mmd_confidence_gate", "entropy",
    "--lambda_max", "0.2",
    "--lambda_min", "0.05",
    "--resgca_geo_tau", "1.0",
    "--resgca_geo_weight", "1.0",
    "--proto_tau", "0.07",
    "--fusion_tau", "0.5",
    "--topk", "8",
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

python @RunArgs
