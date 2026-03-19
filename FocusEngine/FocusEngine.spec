# -*- mode: python ; coding: utf-8 -*-


import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden_imports = ['pyautogui', 'psutil', 'pystray', 'PIL', 'dotenv', 'accelerate']
hidden_imports += collect_submodules('transformers')
hidden_imports += collect_submodules('openvino')
hidden_imports += collect_submodules('optimum')

datas = [('.env', '.')] if os.path.exists('.env') else []
datas += collect_data_files('transformers')
datas += collect_data_files('openvino')
datas += collect_data_files('optimum')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    module_collection_mode={
        'transformers': 'py',
        'optimum': 'py',
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FocusEngine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FocusEngine',
)
