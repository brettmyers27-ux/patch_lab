[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $InstallScript,
    [Parameter(Mandatory)] [string] $SourceBundle,
    [string] $InstallRoot = ""
)

$ErrorActionPreference = "Stop"
$logDirectory = Join-Path $env:LOCALAPPDATA "Patch Lab"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "installer-bootstrap.log"
Start-Transcript -LiteralPath $logPath -Append | Out-Null
trap {
    Write-Error $_
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}

function Refresh-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = @($machine, $user) -join ";"
}

function Test-Python311 {
    foreach ($candidate in @("py.exe", "python3.11.exe", "python.exe")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        $series = if ($candidate -eq "py.exe") {
            & $command.Source -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        } else {
            & $command.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        }
        if ($LASTEXITCODE -eq 0 -and $series -eq "3.11") { return $true }
    }
    return $false
}

function Install-WingetPackage([string] $Id, [string] $Label) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$Label is required and Windows Package Manager is unavailable."
    }
    & $winget.Source install --id $Id --exact --silent --scope user `
        --accept-package-agreements --accept-source-agreements `
        --disable-interactivity
    if ($LASTEXITCODE -ne 0) { throw "Automatic $Label installation failed." }
    Refresh-ProcessPath
}

if (-not (Test-Python311)) {
    Install-WingetPackage "Python.Python.3.11" "Python 3.11"
}
if (-not (Test-Python311)) { throw "Python 3.11 is still unavailable after installation." }

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Install-WingetPackage "Git.Git" "Git for Windows"
}
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    throw "Git for Windows is still unavailable after installation."
}

$env:PATCHLAB_REPO_BUNDLE = [IO.Path]::GetFullPath($SourceBundle)
if ($InstallRoot) { $env:PATCHLAB_INSTALL_ROOT = [IO.Path]::GetFullPath($InstallRoot) }

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallScript
if ($LASTEXITCODE -ne 0) { throw "PatchLab installation failed with exit code $LASTEXITCODE." }
Stop-Transcript | Out-Null
