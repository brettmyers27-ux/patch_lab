# PatchLab trusted-group installer for Windows 11 x64.
#
# Safe to rerun: source updates are fast-forward only, dependencies use a
# checksum marker, and every large download resumes through a .part file.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoUrl = if ($env:PATCHLAB_REPO_URL) { $env:PATCHLAB_REPO_URL } else { "https://github.com/brettmyers27-ux/patch_lab.git" }
$RelayUrl = if ($env:PATCHLAB_RELAY_URL) { $env:PATCHLAB_RELAY_URL } else { "https://patchlab-relay-482507024870.us-central1.run.app" }
$InstallRoot = if ($env:PATCHLAB_INSTALL_ROOT) { $env:PATCHLAB_INSTALL_ROOT } else { Join-Path ([Environment]::GetFolderPath("MyDocuments")) "PatchLab\soundmatch" }
$TestMode = $env:PATCHLAB_INSTALL_TEST_MODE -eq "1"

function Write-Step([string]$Message) {
    Write-Host $Message
}

function Stop-Install([string]$Message) {
    Write-Error "PatchLab install error: $Message"
    exit 1
}

function Get-Python311 {
    $attempts = [System.Collections.Generic.List[string]]::new()
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $attempts.Add("py.exe -3.11")
        $path = & py.exe -3.11 -c "import sys; print(sys.executable)" 2>$null
        $series = & py.exe -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $series -eq "3.11" -and $path) {
            return ([string]$path).Trim()
        }
    }
    foreach ($name in @("python3.11.exe", "python.exe")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        $attempts.Add([string]$command.Source)
        if ([string]$command.Source -match "\\WindowsApps\\") {
            continue
        }
        $path = & $command.Source -c "import sys; print(sys.executable)" 2>$null
        $series = & $command.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $series -eq "3.11" -and $path) {
            $resolved = ([string]$path).Trim()
            if ($resolved -notmatch "\\WindowsApps\\") {
                return $resolved
            }
        }
    }
    $detail = if ($attempts.Count) { $attempts -join ", " } else { "no Python command was found" }
    Stop-Install "Python 3.11 is missing or unusable ($detail). Install the 64-bit Python 3.11 release from python.org, enable 'Add python.exe to PATH', disable the Microsoft Store App execution aliases for python.exe/python3.exe, then reopen PowerShell."
}

function Get-Vst2Roots {
    $roots = [System.Collections.Generic.List[string]]::new()
    foreach ($registryPath in @(
        "HKLM:\SOFTWARE\VST",
        "HKCU:\SOFTWARE\VST",
        "HKLM:\SOFTWARE\WOW6432Node\VST",
        "HKCU:\SOFTWARE\WOW6432Node\VST"
    )) {
        try {
            $value = (Get-ItemProperty -LiteralPath $registryPath -Name VSTPluginsPath -ErrorAction Stop).VSTPluginsPath
            if ($value) { $roots.Add([Environment]::ExpandEnvironmentVariables(([string]$value).Trim('"'))) }
        } catch {
            # Missing registry keys are normal; common folders are checked below.
        }
    }
    $programFiles64 = if (${env:ProgramW6432}) { ${env:ProgramW6432} } else { $env:ProgramFiles }
    $roots.Add((Join-Path $programFiles64 "Common Files\VST2"))
    $roots.Add((Join-Path $programFiles64 "VSTPlugins"))
    $roots.Add((Join-Path $programFiles64 "Steinberg\VSTPlugins"))
    return @($roots | Select-Object -Unique)
}

function Get-PluginPreflight {
    $programFiles64 = if (${env:ProgramW6432}) { ${env:ProgramW6432} } else { $env:ProgramFiles }
    $serum1 = [System.Collections.Generic.List[string]]::new()
    foreach ($root in Get-Vst2Roots) {
        $serum1.Add((Join-Path $root "Serum_x64.dll"))
        $serum1.Add((Join-Path $root "Serum.dll"))
    }
    $serum2 = @(
        (Join-Path $programFiles64 "Common Files\VST3\Serum2.vst3")
    )
    return [PSCustomObject]@{
        Serum1Search = @($serum1 | Select-Object -Unique)
        Serum2Search = $serum2
        Serum1 = @($serum1 | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
        Serum2 = @($serum2 | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
    }
}

function Get-TorchChoice {
    if ($env:PATCHLAB_WINDOWS_TORCH -in @("cpu", "cuda")) {
        return [PSCustomObject]@{
            Flavor = $env:PATCHLAB_WINDOWS_TORCH
            Reason = "PATCHLAB_WINDOWS_TORCH=$($env:PATCHLAB_WINDOWS_TORCH) override"
        }
    }
    $names = @()
    try {
        $names = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name })
    } catch {
        Write-Warning "Could not query display adapters through Windows CIM; selecting CPU PyTorch. Set PATCHLAB_WINDOWS_TORCH=cuda and rerun if this machine has a supported NVIDIA GPU."
    }
    $nvidia = @($names | Where-Object { $_ -match "NVIDIA" })
    if ($nvidia.Count) {
        return [PSCustomObject]@{
            Flavor = "cuda"
            Reason = "NVIDIA adapter detected: $($nvidia -join ', '); selecting CUDA 12.8 wheels"
        }
    }
    return [PSCustomObject]@{
        Flavor = "cpu"
        Reason = "No NVIDIA adapter detected; selecting smaller CPU-only PyTorch wheels"
    }
}

function Invoke-SecretAuth([string]$Python, [string]$Script, [string]$Url, [Security.SecureString]$Secret) {
    $pointer = [IntPtr]::Zero
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $Python
        $quotedScript = '"' + $Script.Replace('"', '\"') + '"'
        $quotedUrl = '"' + $Url.Replace('"', '\"') + '"'
        $info.Arguments = "-u $quotedScript auth --relay-url $quotedUrl"
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardInput = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = [Diagnostics.Process]::Start($info)
        $process.StandardInput.Write($plain)
        $process.StandardInput.Close()
        $output = $process.StandardOutput.ReadToEnd()
        $errorOutput = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($output) { Write-Host $output.TrimEnd() }
        if ($process.ExitCode -ne 0) {
            if ($errorOutput) { Write-Error $errorOutput.TrimEnd() }
            Stop-Install "The private-group passcode was not accepted or the relay was unavailable."
        }
    } finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
        Remove-Variable plain -ErrorAction SilentlyContinue
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    Stop-Install "This installer requires Windows 11."
}
$windowsBuild = [Environment]::OSVersion.Version.Build
if ($windowsBuild -lt 22000) {
    Stop-Install "PatchLab requires Windows 11 (build 22000 or newer); this machine reports build $windowsBuild."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    Stop-Install "PatchLab requires 64-bit Windows 11."
}
if (-not $env:ProgramFiles) {
    Stop-Install "The Program Files environment is unavailable; the Windows installation cannot be validated."
}
$Python = Get-Python311
$PythonVersion = (& $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Stop-Install "Git is missing. Install Git for Windows from https://git-scm.com/download/win, reopen PowerShell, and rerun."
}

$installFullPath = [IO.Path]::GetFullPath($InstallRoot)
$installDriveRoot = [IO.Path]::GetPathRoot($installFullPath)
$drive = [IO.DriveInfo]::new($installDriveRoot)
$requiredBytes = 8GB
if ($drive.AvailableFreeSpace -lt $requiredBytes) {
    $freeGb = [Math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    Stop-Install "At least 8 GB free is required on $installDriveRoot; only $freeGb GB is available."
}

$plugins = Get-PluginPreflight
if (-not $plugins.Serum1.Count) {
    Write-Host "Serum 1 VST2 was not found. Every location searched:"
    $plugins.Serum1Search | ForEach-Object { Write-Host "  - $_" }
    Stop-Install "Install the licensed 64-bit Serum 1 VST2 plug-in (Serum_x64.dll), or set its VSTPluginsPath registry value, then rerun."
}
if (-not $plugins.Serum2.Count) {
    Write-Host "Serum 2 VST3 was not found. Every location searched:"
    $plugins.Serum2Search | ForEach-Object { Write-Host "  - $_" }
    Stop-Install "Install the licensed Serum 2 VST3 instrument, then rerun."
}
$torch = Get-TorchChoice
Write-Step "PatchLab preflight passed: Windows build $windowsBuild, x64, Python $PythonVersion, at least 8 GB free."
Write-Step "  Serum 1 VST2: $($plugins.Serum1[0])"
Write-Step "  Serum 2 VST3: $($plugins.Serum2[0])"
Write-Step "  PyTorch: $($torch.Reason)"

if (Test-Path -LiteralPath $InstallRoot) {
    if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot ".git"))) {
        Stop-Install "$InstallRoot exists but is not a Git checkout. It was left untouched."
    }
    $dirty = & git.exe -C $InstallRoot status --porcelain
    if ($dirty) {
        Stop-Install "The existing checkout has local changes. They were left untouched; commit or move them before rerunning."
    }
    Write-Step "Updating existing PatchLab checkout (fast-forward only)..."
    & git.exe -C $InstallRoot pull --ff-only
    if ($LASTEXITCODE -ne 0) { Stop-Install "The checkout cannot fast-forward. It was left untouched." }
} else {
    Write-Step "Cloning the PatchLab source..."
    $parent = Split-Path -Parent $InstallRoot
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    & git.exe clone $RepoUrl $InstallRoot
    if ($LASTEXITCODE -ne 0) { Stop-Install "Could not clone the PatchLab repository." }
}

Set-Location $InstallRoot
$VenvPython = Join-Path $InstallRoot ".venv\Scripts\python.exe"
$VenvPythonw = Join-Path $InstallRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Step "Creating the Python 3.11 environment..."
    & $Python -m venv (Join-Path $InstallRoot ".venv")
    if ($LASTEXITCODE -ne 0) { Stop-Install "Could not create the virtual environment." }
}
if (-not (Test-Path -LiteralPath $VenvPythonw)) {
    Stop-Install "The Python environment has no pythonw.exe; reinstall 64-bit Python 3.11 from python.org rather than the Microsoft Store."
}

$requirementsHash = (Get-FileHash -Algorithm SHA256 (Join-Path $InstallRoot "requirements.txt")).Hash.ToLowerInvariant()
$hashInput = [Text.Encoding]::UTF8.GetBytes("$requirementsHash|windows-$($torch.Flavor)-torch-v2")
$hasher = [Security.Cryptography.SHA256]::Create()
$dependencyHash = ([BitConverter]::ToString($hasher.ComputeHash($hashInput))).Replace("-", "").ToLowerInvariant()
$hasher.Dispose()
$dependencyMarker = Join-Path $InstallRoot ".venv\.patchlab-dependencies-$dependencyHash"
if (-not (Test-Path -LiteralPath $dependencyMarker)) {
    Write-Step "Installing PatchLab dependencies. This is the longest setup step..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Stop-Install "pip could not update." }
    $torchWheel = if ($torch.Flavor -eq "cuda") { "cu128" } else { "cpu" }
    $torchIndex = "https://download.pytorch.org/whl/$torchWheel"
    & $VenvPython -m pip install torch torchaudio torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { Stop-Install "PyTorch installation failed for $($torch.Flavor): $torchIndex" }
    & $VenvPython -m pip install -r (Join-Path $InstallRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Stop-Install "PatchLab dependency installation failed." }
    Get-ChildItem -LiteralPath (Join-Path $InstallRoot ".venv") -Filter ".patchlab-dependencies-*" -File | Remove-Item -Force
    New-Item -ItemType File -Path $dependencyMarker -Force | Out-Null
} else {
    Write-Step "Dependencies already match this checkout; skipping installation."
}

$env:PATCHLAB_RELAY_URL = $RelayUrl
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $VenvPython (Join-Path $InstallRoot "scripts\install_support.py") auth-status 1>$null 2>$null
$authStatusExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($authStatusExitCode -eq 0) {
    Write-Step "Existing group authentication found; the app will not prompt again."
} else {
    if ($TestMode -and $env:PATCHLAB_PASSCODE_FILE) {
        $plainTestPasscode = (Get-Content -LiteralPath $env:PATCHLAB_PASSCODE_FILE -Raw).TrimEnd("`r", "`n")
        $securePasscode = ConvertTo-SecureString $plainTestPasscode -AsPlainText -Force
        Remove-Variable plainTestPasscode -ErrorAction SilentlyContinue
    } else {
        $securePasscode = Read-Host "Private-group passcode (input hidden)" -AsSecureString
    }
    Invoke-SecretAuth $VenvPython (Join-Path $InstallRoot "scripts\install_support.py") $RelayUrl $securePasscode
    Remove-Variable securePasscode -ErrorAction SilentlyContinue
}

& $VenvPython (Join-Path $InstallRoot "scripts\install_support.py") artifacts-preflight --relay-url $RelayUrl
if ($LASTEXITCODE -ne 0) { Stop-Install "Private artifacts are not reachable. No multi-gigabyte CLAP download was started." }
& $VenvPython (Join-Path $InstallRoot "scripts\install_support.py") clap --install-root $InstallRoot
if ($LASTEXITCODE -ne 0) { Stop-Install "CLAP checkpoint download failed; completed bytes were preserved for a retry." }
$clapMarker = Join-Path $InstallRoot "data\models\huggingface\.patchlab-clap-runtime-v1"
if (-not (Test-Path -LiteralPath $clapMarker)) {
    Write-Step "Preparing CLAP runtime files for offline first use..."
    & $VenvPython (Join-Path $InstallRoot "scripts\cache_clap.py")
    if ($LASTEXITCODE -ne 0) { Stop-Install "CLAP runtime preparation failed; rerun to reuse completed downloads." }
    New-Item -ItemType File -Path $clapMarker -Force | Out-Null
}
& $VenvPython (Join-Path $InstallRoot "scripts\install_support.py") artifacts --relay-url $RelayUrl --install-root $InstallRoot
if ($LASTEXITCODE -ne 0) { Stop-Install "Private artifact download failed; completed bytes were preserved for a retry." }

Write-Step "Running the Windows parity gate before installing shortcuts..."
$env:PATCHLAB_DISTRIBUTION_MODE = "1"
$env:PATCHLAB_MODEL_CACHE = Join-Path $InstallRoot "data\models\huggingface"
& $VenvPython (Join-Path $InstallRoot "scripts\verify_windows_install.py") --installer-gate
if ($LASTEXITCODE -ne 0) {
    Stop-Install "The Windows parity gate failed. No launch shortcuts were created; copy the diagnostic table above back to the PatchLab maintainer."
}

$launcherConfig = @{
    relay_url = $RelayUrl
    model_cache = (Join-Path $InstallRoot "data\models\huggingface")
} | ConvertTo-Json
Set-Content -LiteralPath (Join-Path $InstallRoot ".patchlab-launcher.json") -Value $launcherConfig -Encoding UTF8
$icon = Join-Path $InstallRoot "app\icons\PatchLab.ico"
$launcher = Join-Path $InstallRoot "app\windows_launcher.pyw"
$desktop = if ($env:PATCHLAB_DESKTOP_DIR) { $env:PATCHLAB_DESKTOP_DIR } else { [Environment]::GetFolderPath("Desktop") }
$programs = if ($env:PATCHLAB_STARTMENU_DIR) { $env:PATCHLAB_STARTMENU_DIR } else { [Environment]::GetFolderPath("Programs") }
$startFolder = Join-Path $programs "PatchLab"
New-Item -ItemType Directory -Path $desktop -Force | Out-Null
New-Item -ItemType Directory -Path $startFolder -Force | Out-Null
$shell = New-Object -ComObject WScript.Shell
foreach ($shortcutPath in @((Join-Path $desktop "PatchLab.lnk"), (Join-Path $startFolder "PatchLab.lnk"))) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $VenvPythonw
    $shortcut.Arguments = '"' + $launcher + '"'
    $shortcut.WorkingDirectory = $InstallRoot
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = "PatchLab"
    $shortcut.Save()
}

$diskBytes = (Get-ChildItem -LiteralPath $InstallRoot -File -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$diskGb = [Math]::Round($diskBytes / 1GB, 2)
Write-Host ""
Write-Host "PatchLab installation complete."
Write-Host "  Source and models: $InstallRoot"
Write-Host "  Desktop shortcut:  $(Join-Path $desktop 'PatchLab.lnk')"
Write-Host "  Start Menu:         $(Join-Path $startFolder 'PatchLab.lnk')"
Write-Host "  Disk used:          $diskGb GB"
Write-Host "Launch PatchLab from the Desktop or Start Menu; no terminal will appear."
Write-Host "If Microsoft Defender SmartScreen appears, choose More info, verify the PatchLab source, then choose Run anyway."
Write-Host "To rerun the copy-pasteable Windows diagnostic:"
Write-Host "  & `"$VenvPython`" `"$InstallRoot\scripts\verify_windows_install.py`""
