; Focus Engine Installer Script
[Setup]
AppName=Focus Engine
AppVersion=1.0
DefaultDirName={autopf}\FocusEngine
DefaultGroupName=Focus Engine
OutputDir=.\InstallerOutput
OutputBaseFilename=FocusEngine_Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startup"; Description: "Launch Focus Engine on Windows Startup"; GroupDescription: "Additional options:"; Flags: unchecked

[Files]
Source: "dist\FocusEngine\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README_INSTALL.txt"; DestDir: "{app}"; Flags: isreadme

[Icons]
Name: "{group}\Focus Engine"; Filename: "{app}\FocusEngine.exe"
Name: "{commondesktop}\Focus Engine"; Filename: "{app}\FocusEngine.exe"; Tasks: desktopicon
Name: "{group}\{cm:UninstallProgram,Focus Engine}"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "FocusEngine"; ValueData: """{app}\FocusEngine.exe"""; Flags: uninsdeletevalue; Tasks: startup

[UninstallDelete]
; Clean up configuration and logs in %LOCALAPPDATA%
Type: filesandordirs; Name: "{localappdata}\FocusEngine"

[Run]
Filename: "{app}\FocusEngine.exe"; Description: "{cm:LaunchProgram,Focus Engine}"; Flags: nowait postinstall skipifsilent
