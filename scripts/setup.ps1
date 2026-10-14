param([switch]$SkipModelSmoke)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    $env:UV_HTTP_TIMEOUT = '600'
    & uv sync --locked --group dev
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    $checkArguments = @('run', '--no-sync', 'python', '-m', 'src.cli', 'check-environment')
    if (-not $SkipModelSmoke) { $checkArguments += '--smoke-models' }
    & uv @checkArguments
    if ($LASTEXITCODE -ne 0) { throw 'Environment smoke check failed. Inspect results/phase1/environment.json.' }
}
finally {
    Pop-Location
}
