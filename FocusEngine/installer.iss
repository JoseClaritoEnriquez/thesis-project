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

[Files]
; Grab everything from the PyInstaller Dist Folder
Source: "dist\FocusEngine\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Focus Engine"; Filename: "{app}\FocusEngine.exe"
Name: "{commondesktop}\Focus Engine"; Filename: "{app}\FocusEngine.exe"; Tasks: desktopicon
Name: "{group}\{cm:UninstallProgram,Focus Engine}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\FocusEngine.exe"; Description: "{cm:LaunchProgram,Focus Engine}"; Flags: nowait postinstall skipifsilent
