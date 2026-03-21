# Focus Engine

A system monitoring and AI-powered focus enhancement tool.

## Build and Installation

### Prerequisites
- Python 3.10+
- Inno Setup (for creating the installer)

### 1. Build the Executable
To bundle the application and its AI dependencies:
```powershell
pip install pyinstaller
python -m PyInstaller FocusEngine.spec
```

### 2. Create the Installer
Use Inno Setup to compile `FocusEngine/installer.iss`. The output will be in `FocusEngine/InstallerOutput`.
