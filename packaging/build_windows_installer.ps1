[CmdletBinding()]
param(
    [string] $OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) {
    $OutputRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "..\releases\v1\windows"))
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

if ((git -C $projectRoot status --porcelain)) {
    throw "The V1 source tree must be clean before packaging."
}
if ((git -C $projectRoot branch --show-current) -ne "main") {
    throw "Build the public release from V1 main."
}

$versionLine = Get-Content -LiteralPath (Join-Path $projectRoot "app\__version__.py") |
    Where-Object { $_ -match '^__version__\s*=\s*"([^"]+)"' } |
    Select-Object -First 1
if (-not $versionLine -or $versionLine -notmatch '"([^"]+)"') {
    throw "Could not read the PatchLab version."
}
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
$sensitiveMatches = git -C $projectRoot grep -I -n -E $credentialPattern `
    HEAD -- . ':(exclude)tests/**' 2>$null
if ($LASTEXITCODE -eq 0 -and $sensitiveMatches) {
    throw "Potential credential material found in tracked release source."
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("patchlab-win-build-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    $sourceBundle = Join-Path $tempRoot "PatchLab-source.bundle"
    & git -C $projectRoot bundle create $sourceBundle main
    if ($LASTEXITCODE -ne 0) { throw "Could not create the public source bundle." }
    & git -C $projectRoot bundle verify $sourceBundle | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "The public source bundle is invalid." }

    $compilerCandidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $compiler = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $compiler) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) { throw "Inno Setup is absent and Windows Package Manager is unavailable." }
        & $winget.Source install --id JRSoftware.InnoSetup --exact --silent --scope user `
            --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -ne 0) { throw "Automatic Inno Setup installation failed." }
        $compiler = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }
    if (-not $compiler) { throw "Inno Setup compiler was not found after installation." }

    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    & $compiler "/DAppVersion=$version" "/DSourceBundle=$sourceBundle" `
        "/DOutputDir=$OutputRoot" "/DProjectRoot=$projectRoot" `
        (Join-Path $PSScriptRoot "windows\PatchLab.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

    $exe = Join-Path $OutputRoot "PatchLab-v$version-windows-x64.exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "Expected installer was not produced: $exe" }
    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$exe.sha256" -Encoding ascii -NoNewline `
        -Value "$sha256  $([IO.Path]::GetFileName($exe))`n"
    $manifest = [ordered]@{
        patchlab_version = $version
        source_commit = $sourceCommit
        build_date_utc = [DateTime]::UtcNow.ToString("o")
        architecture = "windows-x64"
        installer_technology = "Inno Setup 6 thin bootstrapper"
        runtime_family_id = "v1-legacy-stock-clap"
        external_prerequisites = @(
            "Windows 11 x64",
            "licensed Serum 1 VST2 and Serum 2 VST3",
            "internet access",
            "PatchLab trusted-group passcode"
        )
        installer_sha256 = $sha256
        signed = $false
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputRoot "release-manifest.json") -Encoding utf8
    @"
# PatchLab V1 Windows release

Version: $version
Source commit: $sourceCommit
Architecture: Windows x64
Installer: Inno Setup 6 thin bootstrapper around the canonical install.ps1 flow
Runtime family: v1-legacy-stock-clap
SHA-256: $sha256

Requirements: Windows 11 x64, licensed Serum 1 VST2 and Serum 2 VST3,
internet access, and the PatchLab trusted-group passcode. The installer is
unsigned, so Windows SmartScreen may display a warning.
"@ | Set-Content -LiteralPath (Join-Path $OutputRoot "README.md") -Encoding utf8

    Write-Host "WINDOWS_INSTALLER=$exe"
    Write-Host "WINDOWS_INSTALLER_SHA256=$sha256"
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
