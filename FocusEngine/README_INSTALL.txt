Focus Engine Installation Notes
==============================

Dependencies
============

Required software:
- Python 3.11 (recommended for this project)
- Microsoft Visual C++ Redistributable (x64):
  https://aka.ms/vs/17/release/vc_redist.x64.exe
- Inno Setup (only needed to build the installer)

Python packages (from requirements.txt):
- pyautogui
- pillow
- psutil
- pystray
- huggingface_hub
- numpy
- python-dotenv

llama-cpp-python install profile:
- jamepeng / jamepeng-cpu (source install from JamePeng fork)
- nvidia (CUDA build)
- amd (Vulkan build)
- jamepeng-nvidia (fork + CUDA CMake args)
- jamepeng-amd (fork + Vulkan CMake args)
- jamepeng-cu126 (default, prebuilt wheel for CUDA 12.6, v0.3.33)
- cpu (prebuilt wheel fallback)

Compatibility note:
- Focus Engine uses whichever compatible Qwen VL chat handler is available
  (`Qwen25VLChatHandler`, `Qwen2VLChatHandler`, then `Qwen3VLChatHandler`).
- The default `cpu` profile may miss newer handlers/features.
- Use `jamepeng*` profiles to build from `https://github.com/JamePeng/llama-cpp-python` when you need newer llama-cpp-python capabilities.

Default model assets:
- Model target: Qwen2.5-VL-3B
- Hugging Face repo: unsloth/Qwen2.5-VL-3B-Instruct-GGUF
- Required files:
  - Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
  - mmproj-F16.gguf
- Runtime/download defaults are defined in focus_engine.py and mirrored in download_model.py.

Startup Procedure
=================

Run from source:
1. Open terminal in FocusEngine folder.
2. Create and activate the project virtual environment:
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
3. Install dependencies for your profile:
   python install_runtime.py
   (Use --profile nvidia or --profile amd for legacy placeholders.)
   To use the JamePeng fork:
   python install_runtime.py --profile jamepeng
   For fork GPU variants:
   python install_runtime.py --profile jamepeng-nvidia
   python install_runtime.py --profile jamepeng-amd
   If you previously installed a different `llama-cpp-python` version, rerun the command with your chosen profile.
4. Start the app:
   python main.py
5. On first launch, select your local GGUF model folder when prompted.
6. The selected model path is saved to config.json for future launches.

Run packaged app (.exe):
1. Install Microsoft Visual C++ Redistributable (x64) if not already installed.
2. Start FocusEngine.exe.
3. On first launch, select your model folder when prompted.
4. Logs and runtime config are saved under %LOCALAPPDATA%\FocusEngine.

If the application fails to start with a "DLL initialization routine failed" or "WinError 1114" error:

1. Install the latest Microsoft Visual C++ Redistributable (x64):
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Restart your computer.

The Focus Engine requires these runtime libraries to initialize its AI components (llama.cpp backend).
