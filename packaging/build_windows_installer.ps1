[CmdletBinding()]
param(
    [string] $OutputRoot = "",
    [Parameter(Mandatory)] [string] $Python
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) {
    $OutputRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "..\releases\v1\windows"))
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$Python = [IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python)) { throw "Build Python was not found: $Python" }

$pythonVersion = (& $Python -c "import platform,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.architecture()[0]}')").Trim()
if ($pythonVersion -ne "3.11|64bit") { throw "Build Python must be 64-bit CPython 3.11; found $pythonVersion." }
if ((git -C $projectRoot status --porcelain)) { throw "The V1 source tree must be clean before packaging." }

$versionLine = Get-Content -LiteralPath (Join-Path $projectRoot "app\__version__.py") |
    Where-Object { $_ -match '^__version__\s*=\s*"([^"]+)' } | Select-Object -First 1
if (-not $versionLine -or $versionLine -notmatch '"([^"]+)"') { throw "Could not read the PatchLab version." }
$version = $Matches[1]
$sourceCommit = (git -C $projectRoot rev-parse HEAD).Trim()

$trackedPaths = @(git -C $projectRoot ls-tree -r --name-only HEAD)
$forbiddenPath = $trackedPaths | Where-Object {
    $_ -match '(^|/)(data|private|\.venv|gcloud|relay-credentials)(/|$)' -or
    $_ -match '(?i)(service.account|private.key|credentials\.json|\.pfx$|\.pem$)'
} | Select-Object -First 1
if ($forbiddenPath) { throw "Sensitive path is tracked and cannot be packaged: $forbiddenPath" }
$credentialPattern = '(BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|refresh_' +
    'token["'']*[:=]|client_' + 'secret["'']*[:=])'
$sensitiveMatches = git -C $projectRoot grep -I -n -E $credentialPattern HEAD -- . ':(exclude)tests/**' 2>$null
if ($LASTEXITCODE -eq 0 -and $sensitiveMatches) { throw "Potential credential material found in tracked release source." }

$compilerCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$compiler = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $compiler) { throw "Inno Setup 6 is required to build a Windows release candidate." }

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("patchlab-win-build-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    $freezeDist = Join-Path $tempRoot "dist"
    $freezeWork = Join-Path $tempRoot "pyinstaller-work"
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $tempRoot "pyinstaller-cache"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    & $Python -m PyInstaller --clean --noconfirm --distpath $freezeDist --workpath $freezeWork (Join-Path $projectRoot "packaging\patchlab.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
    $payload = Join-Path $freezeDist "PatchLab"
    $appExe = Join-Path $payload "PatchLab.exe"
    if (-not (Test-Path -LiteralPath $appExe)) { throw "PyInstaller did not produce PatchLab.exe." }

    & $appExe --patchlab-worker packaged-runtime-gate --expected-commit $sourceCommit
    if ($LASTEXITCODE -ne 0) { throw "Frozen runtime gate failed with exit code $LASTEXITCODE." }

    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    & $compiler "/DAppVersion=$version" "/DAppPayload=$payload" `
        "/DOutputDir=$OutputRoot" "/DProjectRoot=$projectRoot" `
        (Join-Path $PSScriptRoot "windows\PatchLab.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

    $exe = Join-Path $OutputRoot "PatchLab-v$version-windows-x64.exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "Expected installer was not produced: $exe" }
    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$exe.sha256" -Encoding ascii -NoNewline -Value "$sha256  $([IO.Path]::GetFileName($exe))`n"
    [ordered]@{
        patchlab_version = $version; source_commit = $sourceCommit
        build_date_utc = [DateTime]::UtcNow.ToString("o"); architecture = "windows-x64"
        installer_technology = "Inno Setup 6 frozen-payload installer"
        runtime_family_id = "v1-legacy-stock-clap"; minimum_windows = "Windows 10 x64 build 19041"
        external_prerequisites = @("licensed Serum installation", "internet access", "PatchLab trusted-group passcode")
        installer_sha256 = $sha256; signed = $false
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputRoot "release-manifest.json") -Encoding utf8
    Write-Host "WINDOWS_INSTALLER=$exe"
    Write-Host "WINDOWS_INSTALLER_SHA256=$sha256"
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
