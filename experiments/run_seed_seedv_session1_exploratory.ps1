param(
    [ValidateSet("Smoke", "SingleSessionLOSO")]
    [string]$Stage = "Smoke",
    [string[]]$Datasets = @("seed", "seedv"),
    [int]$Seed = 42,
    [string]$Python = $env:SGDA_PYTHON
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = "python"
}

$Runner = "experiments\crossSubject_geosem_stda_sgda.py"
foreach ($Dataset in $Datasets) {
    $Dataset = $Dataset.ToLowerInvariant()
    if ($Dataset -notin @("seed", "seedv")) {
        throw "Unsupported dataset '$Dataset'; choose seed and/or seedv."
    }

    # Requested exploratory scope: SEED Session 2 and SEED-V Session 1.
    $Session = if ($Dataset -eq "seed") { 2 } else { 1 }

    if ($Stage -eq "Smoke") {
        $Epochs = 2
        $Targets = @("1")
    }
    else {
        $Epochs = 200
        if ($Dataset -eq "seed") {
            $Targets = 1..15 | ForEach-Object { [string]$_ }
        }
        else {
            $Targets = 1..16 | ForEach-Object { [string]$_ }
        }
    }

    $LaunchArguments = @(
        $Runner,
        "--dataset_name", $Dataset,
        "--epochs", [string]$Epochs,
        "--batch_size", "64",
        "--lr", "1e-3",
        "--sample_length", "3",
        "--stride", "1",
        "--seed", [string]$Seed,
        "--session_ids", [string]$Session,
        "--target_subject_ids"
    ) + $Targets + @(
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

    Write-Host "Starting $Stage for $Dataset, session $Session, seed $Seed, targets: $($Targets -join ',')"
    & $Python @LaunchArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Dataset run failed with exit code $LASTEXITCODE. See the runner's per-run training log."
    }
}

Write-Host "Requested exploratory run(s) completed. Inspect results/results_<dataset>_geosem_stda/runs and do not pool SEED with SEED-V metrics."
