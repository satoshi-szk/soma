# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(SPECPATH).resolve()
frontend_dist = project_root / "frontend" / "dist"

datas = []
if frontend_dist.exists():
    datas.append((str(frontend_dist), "frontend/dist"))

block_cipher = None

a = Analysis(
    ["src/soma/__main__.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="soma",
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="soma",
)

app = BUNDLE(
    coll,
    name="SOMA.app",
    icon=None,
    bundle_identifier="com.soma.app",
    info_plist={
        "CFBundleName": "SOMA",
        "CFBundleDisplayName": "SOMA",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "SOMA needs microphone access for audio analysis.",
    },
)
