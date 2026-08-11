param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
  & $Python run_experiment.py `
    --config configs\mock.toml `
    --data data\sample_who_when.jsonl `
    --methods who_when_official_all_at_once,a2p_repo_exact,ccv_ablation_no_gt,mvbs10 `
    --out outputs\smoke `
    --repeats 1

  if ($LASTEXITCODE -ne 0) {
    throw "Smoke experiment failed with exit code $LASTEXITCODE"
  }

  & $Python -m unittest discover -s tests -v
  if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}
