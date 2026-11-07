param(
    [ValidateSet('status', 'stop', 'stop-demos')]
    [string]$Action = 'status'
)

$snakeRoot = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = 'Stop'
$snakeSessionPath = Join-Path $snakeRoot 'sessions\current.json'
if (-not (Test-Path -LiteralPath $snakeSessionPath)) {
    throw 'No background training session has been recorded.'
}
$snakeSession = Get-Content -LiteralPath $snakeSessionPath -Raw -Encoding UTF8 | ConvertFrom-Json
$snakeRunDirectory = [IO.Path]::GetFullPath($snakeSession.RunDirectory)
$snakeRunsRoot = [IO.Path]::GetFullPath((Join-Path $snakeRoot 'runs')) + [IO.Path]::DirectorySeparatorChar
if (-not $snakeRunDirectory.StartsWith($snakeRunsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The session run directory is outside this project.'
}
switch ($Action) {
    'status' {
        Get-Content -LiteralPath (Join-Path $snakeRunDirectory 'status.json') -Raw -Encoding UTF8
        $snakeLiveStatus = Join-Path $snakeRunDirectory 'live_dashboard_status.json'
        if (Test-Path -LiteralPath $snakeLiveStatus) {
            Get-Content -LiteralPath $snakeLiveStatus -Raw -Encoding UTF8
        }
        $snakeDemoStatus = Join-Path $snakeRunDirectory 'demo_status.json'
        if (Test-Path -LiteralPath $snakeDemoStatus) {
            $snakeDemo = Get-Content -LiteralPath $snakeDemoStatus -Raw -Encoding UTF8 | ConvertFrom-Json
            $snakeElapsed = ([DateTime]::UtcNow - (Get-Item -LiteralPath $snakeDemoStatus).LastWriteTimeUtc).TotalSeconds
            $snakeDemo.seconds_until_next = [Math]::Round([Math]::Max(0, $snakeDemo.seconds_until_next - $snakeElapsed), 1)
            $snakeDemo | ConvertTo-Json
        }
    }
    'stop' {
        New-Item -ItemType File -Path (Join-Path $snakeRunDirectory 'STOP') -Force | Out-Null
        'Training will stop and save after its current iteration.'
    }
    'stop-demos' {
        New-Item -ItemType File -Path (Join-Path $snakeRunDirectory 'STOP_DEMOS') -Force | Out-Null
        'Future demos are disabled. Training continues; the current demo can finish.'
    }
}
