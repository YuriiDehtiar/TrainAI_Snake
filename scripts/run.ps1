param(
    [ValidateSet('train','play','evaluate','dashboard','diagnose_losses','test')]
    [string]$Mode='play',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ScriptArguments
)
$ErrorActionPreference='Stop'
$snakeRoot=Split-Path -Parent $PSScriptRoot
$snakePython=Join-Path $snakeRoot '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $snakePython)){$snakePython=(Get-Command python -ErrorAction Stop).Source}
Push-Location -LiteralPath $snakeRoot
try{
    if($Mode -eq 'test'){& $snakePython -m unittest discover -s tests -v @ScriptArguments}
    else{& $snakePython -u -m "snake_v9.cli.$Mode" @ScriptArguments}
    if($LASTEXITCODE -ne 0){throw "Python exited with code $LASTEXITCODE"}
}finally{Pop-Location}
