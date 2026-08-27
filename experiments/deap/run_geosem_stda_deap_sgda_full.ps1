$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

$RunArgs = @(
    "experiments\crossSubject_geosem_stda_sgda.py",
    "--dataset_name", "deap",
    "--epochs", "200",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--sample_length", "9",
    "--stride", "3",
    "--seed", "42",
    "--source_selection", "sparse_reliability",
    "--reliability_warmup_epochs", "5",
    "--sparse_k_max", "6",
    "--mmd_type", "hut",
    "--mmd_schedule", "warmup_cosine_decay",
    "--mmd_start_ratio", "0.05",
    "--mmd_confidence_gate", "entropy",
    "--lambda_max", "0.02",
    "--lambda_min", "0.0",
    "--uot_epsilon", "0.10",
    "--uot_tau_s", "1.0",
    "--uot_tau_t", "0.7",
    "--uot_route_tau", "0.20",
    "--uot_n_iter", "12",
    "--hut_geo_cost_weight", "0.2",
    "--hut_agreement_tau", "0.5",
    "--no_hut_agreement_mass",
    "--hut_use_geometry_cost",
    "--proto_tau", "0.07",
    "--fusion_tau", "0.5",
    "--topk", "6",
    "--shrinkage", "0.1",
    "--spd_eps", "1e-5",
    "--geometry_batch_size", "256",
    "--st_dim", "128",
    "--graph_dim", "64",
    "--graph_heads", "4",
    "--adapter_bottleneck", "32",
    "--heads", "4",
    "--dropout", "0.3",
    "--weight_decay", "1e-4",
    "--eval_interval", "1",
    "--log_interval", "1",
    "--sample_subset", "stratified"
)

python @RunArgs
