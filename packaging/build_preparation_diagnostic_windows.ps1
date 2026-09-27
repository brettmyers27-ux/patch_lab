param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$OutputDir = "..\release-artifacts\phase4a5-windows"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path ([System.IO.Path]::GetTempPath()) ("patchlab-preparation-diagnostic-" + [guid]::NewGuid())
$Source = Join-Path $Stage "source"
$Dist = Join-Path $Stage "dist"
$Work = Join-Path $Stage "work"

try {
    New-Item -ItemType Directory -Force -Path (Join-Path $Source "core") | Out-Null
    Copy-Item (Join-Path $PSScriptRoot "preparation_diagnostic.py") (Join-Path $Source "preparation_diagnostic.py")
    Copy-Item (Join-Path $Root "core\preparation_diagnostic.py") (Join-Path $Source "core\preparation_diagnostic.py")
    & $Python -m PyInstaller --clean --noconfirm --windowed `
        --name "PatchLab Preparation Diagnostic" `
        --paths $Source --distpath $Dist --workpath $Work `
        (Join-Path $Source "preparation_diagnostic.py")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    $Target = Join-Path (Resolve-Path $OutputDir) "PatchLab Preparation Diagnostic"
    if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
    Copy-Item -Recurse (Join-Path $Dist "PatchLab Preparation Diagnostic") $Target
    Write-Output $Target
}
finally {
    if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
}
