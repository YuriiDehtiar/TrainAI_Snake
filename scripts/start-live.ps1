param(
    [string]$RunName = ('live_fresh_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$Iterations = 2000,
    [string]$ResumeCheckpoint,
    [switch]$DashboardOnly
)

$snakeRoot = Split-Path -Parent $PSScriptRoot

$ErrorActionPreference = 'Stop'
$snakePython = Join-Path $snakeRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $snakePython)) { $snakePython = (Get-Command python -ErrorAction Stop).Source }
$snakePythonW = Join-Path (Split-Path -Parent $snakePython) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $snakePythonW)) { throw 'pythonw.exe is missing.' }
$snakeSessionDirectory = Join-Path $snakeRoot 'sessions'
New-Item -ItemType Directory -Path $snakeSessionDirectory -Force | Out-Null
$snakeSessionPath = Join-Path $snakeSessionDirectory 'current.json'

function Read-SnakeJson([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        try { Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { Write-Warning ('Unreadable JSON after a crash: ' + $Path); return $null }
    }
}

function Write-SnakeSession($Session) {
    $snakeTemporary = $snakeSessionPath + '.tmp'
    $Session | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $snakeTemporary -Encoding UTF8
    Move-Item -LiteralPath $snakeTemporary -Destination $snakeSessionPath -Force
}

$snakePrevious = Read-SnakeJson $snakeSessionPath
if ($DashboardOnly) {
    if (-not $snakePrevious) { throw 'No current session. Start a new run first.' }
    $snakeSession = $snakePrevious
    $snakeRunDirectory = [IO.Path]::GetFullPath($snakeSession.RunDirectory)
    $snakeRunsRoot = [IO.Path]::GetFullPath((Join-Path $snakeRoot 'runs')) + [IO.Path]::DirectorySeparatorChar
    if (-not $snakeRunDirectory.StartsWith($snakeRunsRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The recorded run is outside the project.'
    }
    $snakeDashboardStatus = Read-SnakeJson (Join-Path $snakeRunDirectory 'live_dashboard_status.json')
    if ($snakeDashboardStatus -and (Get-Process -Id $snakeDashboardStatus.pid -ErrorAction SilentlyContinue)) {
        Write-Output 'The dashboard is already running.'
        return
    }
} else {
    if ($Iterations -lt 1) { throw 'Iterations must be positive.' }
    if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw 'Use a simple ASCII directory name.' }
    if ($snakePrevious) {
        $snakePreviousStatus = Read-SnakeJson (Join-Path $snakePrevious.RunDirectory 'status.json')
        if ($snakePreviousStatus -and $snakePreviousStatus.phase -in @('starting', 'training') -and
            (Get-Process -Id $snakePreviousStatus.pid -ErrorAction SilentlyContinue)) {
            throw 'Training is already running. Use -DashboardOnly, or control.ps1 stop first.'
        }
    }
    $snakeRunDirectory = Join-Path $snakeRoot ('runs\' + $RunName)
    if (Test-Path -LiteralPath $snakeRunDirectory) { throw 'The run directory already exists. Choose a new RunName.' }
    $snakeReference = Join-Path $snakeRoot 'configs\training.json'
    if (-not (Test-Path -LiteralPath $snakeReference)) { throw 'The last parameter file is missing.' }
    if ($snakePrevious) {
        Copy-Item -LiteralPath $snakeSessionPath -Destination (Join-Path $snakeSessionDirectory ('before_' + $RunName + '.json'))
    }
    $snakeStdout = Join-Path $snakeSessionDirectory ($RunName + '.stdout.log')
    $snakeStderr = Join-Path $snakeSessionDirectory ($RunName + '.stderr.log')
    $snakeArguments = @('-u', '-m', 'snake_v9.cli.train',
        '--run-name', $RunName, '--settings-from', ('"' + $snakeReference + '"'),
        '--iterations', [string]$Iterations, '--num-envs', '1024', '--horizon', '64',
        '--minibatch-size', '16384', '--epochs', '10', '--seed', '42', '--save-every', '20')
    if ($ResumeCheckpoint) {
        $snakeResumePath = (Resolve-Path -LiteralPath $ResumeCheckpoint).Path
        $snakeArguments += @('--resume', ('"' + $snakeResumePath + '"'))
    }
    $snakeTrainer = Start-Process -FilePath $snakePython -ArgumentList $snakeArguments -WorkingDirectory $snakeRoot `
        -WindowStyle Hidden -RedirectStandardOutput $snakeStdout -RedirectStandardError $snakeStderr -PassThru
    $snakeSession = [pscustomobject][ordered]@{RunDirectory=$snakeRunDirectory; TrainerPID=$snakeTrainer.Id;
        DashboardPID=$null; StandardOutput=$snakeStdout; StandardError=$snakeStderr;
        Started=(Get-Date).ToString('o'); Initialization=$(if ($ResumeCheckpoint) { 'checkpoint' } else { 'random' })}
    Write-SnakeSession $snakeSession
    $snakeDeadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Path -LiteralPath (Join-Path $snakeRunDirectory 'run.json'))) {
        $snakeTrainer.Refresh()
        if ($snakeTrainer.HasExited) { throw ('Training exited. Read ' + $snakeStderr) }
        if ((Get-Date) -gt $snakeDeadline) { throw ('Startup is slow. Read ' + $snakeStderr) }
        Start-Sleep -Milliseconds 250
    }
}

$snakeDashboardArguments = @('-m', 'snake_v9.cli.dashboard',
    '--run-dir', ('"' + $snakeRunDirectory + '"'))
$snakeDashboard = Start-Process -FilePath $snakePythonW -ArgumentList $snakeDashboardArguments -WorkingDirectory $snakeRoot `
    -RedirectStandardOutput ($snakeSession.StandardOutput + '.dashboard') `
    -RedirectStandardError ($snakeSession.StandardError + '.dashboard') -PassThru
$snakeSession.DashboardPID = $snakeDashboard.Id
$snakeCurrentStatus = Read-SnakeJson (Join-Path $snakeRunDirectory 'status.json')
if ($snakeCurrentStatus) { $snakeSession.TrainerPID = $snakeCurrentStatus.pid }
Write-SnakeSession $snakeSession
Write-Output ('Run: ' + $snakeRunDirectory)
Write-Output 'Stop safely: .\scripts\control.ps1 stop'
Write-Output 'Reopen the window: .\scripts\start-live.ps1 -DashboardOnly'
