#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef AppPayload
  #error AppPayload is required
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
DefaultDirName={localappdata}\Programs\PatchLab
CreateAppDir=yes
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=PatchLab-v{#AppVersion}-windows-x64
PrivilegesRequired=lowest
SetupIconFile={#ProjectRoot}\app\icons\PatchLab.ico
SolidCompression=yes
Uninstallable=yes
WizardStyle=modern

[Files]
Source: "{#AppPayload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\PatchLab"; Filename: "{app}\PatchLab.exe"; IconFilename: "{app}\PatchLab.exe"
Name: "{userprograms}\PatchLab\PatchLab"; Filename: "{app}\PatchLab.exe"; IconFilename: "{app}\PatchLab.exe"
