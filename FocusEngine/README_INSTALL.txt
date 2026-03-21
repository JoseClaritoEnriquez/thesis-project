Focus Engine Installation Notes
==============================

If the application fails to start with a "DLL initialization routine failed" or "WinError 1114" error:

1. Install the latest Microsoft Visual C++ Redistributable (x64):
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Restart your computer.

The Focus Engine requires these runtime libraries to initialize its AI components (PyTorch/OpenVINO).
