#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef SourceBundle
  #error SourceBundle is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef ProjectRoot
  #error ProjectRoot is required
#endif

[Setup]
AppId={{D82A620A-B68E-44FA-BC8D-61413B45198D}
AppName=PatchLab
AppVersion={#AppVersion}
AppPublisher=PatchLab
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CreateAppDir=no
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=PatchLab-v{#AppVersion}-windows-x64
PrivilegesRequired=lowest
SetupIconFile={#ProjectRoot}\app\icons\PatchLab.ico
SolidCompression=yes
Uninstallable=no
WizardStyle=modern

[Files]
Source: "{#ProjectRoot}\install.ps1"; DestDir: "{tmp}\PatchLab-bootstrap"; Flags: deleteafterinstall
Source: "{#ProjectRoot}\packaging\windows\bootstrap.ps1"; DestDir: "{tmp}\PatchLab-bootstrap"; Flags: deleteafterinstall
Source: "{#SourceBundle}"; DestDir: "{tmp}\PatchLab-bootstrap"; DestName: "PatchLab-source.bundle"; Flags: deleteafterinstall

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  InstallRoot: String;
  Parameters: String;
begin
  if CurStep <> ssPostInstall then
    exit;

  InstallRoot := ExpandConstant('{param:PATCHLABINSTALLROOT|}');
  Log('PatchLab requested install root: ' + InstallRoot);
  Parameters := '-NoProfile -ExecutionPolicy Bypass -File ' +
    AddQuotes(ExpandConstant('{tmp}\PatchLab-bootstrap\bootstrap.ps1')) +
    ' -InstallScript ' + AddQuotes(ExpandConstant('{tmp}\PatchLab-bootstrap\install.ps1')) +
    ' -SourceBundle ' + AddQuotes(ExpandConstant('{tmp}\PatchLab-bootstrap\PatchLab-source.bundle'));
  if InstallRoot <> '' then
    Parameters := Parameters + ' -InstallRoot ' + AddQuotes(InstallRoot);

  if (not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      Parameters, '', SW_SHOW, ewWaitUntilTerminated, ResultCode)) or
      (ResultCode <> 0) then
    RaiseException('PatchLab bootstrap failed. Review the PowerShell window for details.');
end;
