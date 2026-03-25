# Focus Engine

A system monitoring and AI-powered focus enhancement tool.

## Dependencies

### Required software
- Python 3.11 (recommended)
- Microsoft Visual C++ Redistributable (x64): [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Inno Setup (only needed when creating the installer)

### Python packages
Install project dependencies from `FocusEngine/requirements.txt`:
- `pyautogui`
- `pillow`
- `psutil`
- `pystray`
- `huggingface_hub`
- `numpy`
- `python-dotenv`

`llama-cpp-python` is installed via runtime profile using `FocusEngine/install_runtime.py`.
- `jamepeng` or `jamepeng-cpu` (**default**): source install from [`JamePeng/llama-cpp-python`](https://github.com/JamePeng/llama-cpp-python).
- `jamepeng-nvidia`: source install from fork with `CMAKE_ARGS=-DGGML_CUDA=on`.
- `jamepeng-amd`: source install from fork with `CMAKE_ARGS=-DGGML_VULKAN=on`.
- `cpu`: prebuilt wheel (`llama-cpp-python==0.3.2`) for maximum compatibility/fallback.

Using the fork profiles enables newer upstream features (including newer chat handlers) at the cost of longer install time and local build toolchain requirements.

## Startup Procedure

### Run from source
1. Open a terminal in `FocusEngine`.
2. Create and activate a Python 3.11 virtual environment:
   ```powershell
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   ```
3. Install dependencies for your runtime profile:
   ```powershell
   python install_runtime.py
   ```
   For NVIDIA GPUs:
   ```powershell
   python install_runtime.py --profile nvidia
   ```
   For AMD GPUs:
   ```powershell
   python install_runtime.py --profile amd
   ```
   To use the JamePeng fork instead:
   ```powershell
   python install_runtime.py --profile jamepeng
   ```
   If you already have `llama-cpp-python` installed, rerun one of these commands to reinstall with the selected profile.
4. Start the app:
   ```powershell
   python main.py
   ```
5. On first launch, choose your local GGUF model folder when prompted.
6. The selected model path is saved to `config.json` and reused on next startup.

Default recommendation is `cpu` profile for widest compatibility, with app-level GPU fallback already handled at runtime.
If startup auto-installs dependencies, override profile using `FOCUS_ENGINE_RUNTIME_PROFILE` (for example: `jamepeng`, `jamepeng-nvidia`, `jamepeng-amd`, `cpu`).

### Run packaged app (.exe)
1. Install Microsoft Visual C++ Redistributable (x64) if not already installed.
2. Launch `FocusEngine.exe`.
3. On first launch, select your model folder when prompted.
4. Logs and runtime config are stored in `%LOCALAPPDATA%\FocusEngine`.

## Build and Installation

### Prerequisites
- Python 3.11
- Inno Setup (for creating the installer)

### 1. Build the Executable
To bundle the application and its AI dependencies:
```powershell
.\.venv311\Scripts\Activate.ps1
python -m pip install pyinstaller
python build_exe.py
```

### 2. Create the Installer
Use Inno Setup to compile `FocusEngine/installer.iss`. The output will be in `FocusEngine/InstallerOutput`.
