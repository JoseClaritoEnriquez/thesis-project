import argparse
import os
import shutil
import subprocess
import sys


CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
LLAMA_CPP_VERSION = "0.3.33"
LLAMA_CPP_FORK_SPEC = "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
# Prebuilt CUDA wheel (JamePeng release). Constraints MUST match the filename:
#   Python 3.11 only (cp311), Windows 64-bit (win_amd64), CUDA 12.6-linked build.
# If pip reports "no matching distribution", your interpreter or OS does not match.
JAMEPENG_CU126_WHEEL = "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.33-cu126-Basic-win-20260315/llama_cpp_python-0.3.33+cu126.basic-cp311-cp311-win_amd64.whl"

# MinGW w64devkit: set W64DEVKIT to the kit root (folder that contains `bin/gcc.exe`).
# If unset, we use %USERPROFILE%\Downloads\w64devkit when that directory exists.
_W64DEVKIT_FALLBACK_PARTS = ("Downloads", "w64devkit")


def _win32_openmp_link_cmake_args():
    """GCC/MinGW on Windows: ggml compiles with -fopenmp but may not link libgomp into ggml-cpu.dll."""
    if sys.platform != "win32":
        return ""
    return " ".join(
        [
            "-DCMAKE_SHARED_LINKER_FLAGS=-fopenmp",
            "-DCMAKE_EXE_LINKER_FLAGS=-fopenmp",
        ]
    )


def _cmake_env(*flag_groups):
    parts = [str(p).strip() for p in flag_groups if p and str(p).strip()]
    if not parts:
        return None
    return {"CMAKE_ARGS": " ".join(parts)}


def _require_supported_python_for_source_llama():
    """llama-cpp-python source wheels are validated on Python 3.11 for this project."""
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) >= (3, 13):
        raise RuntimeError(
            f"Source install of llama-cpp-python is not supported on Python {major}.{minor} here. "
            "Activate FocusEngine/.venv311 (Python 3.11), then run install_runtime.py again."
        )


def _w64devkit_fallback_root():
    root = os.path.join(os.path.expanduser("~"), *_W64DEVKIT_FALLBACK_PARTS)
    return os.path.normpath(root) if os.path.isdir(root) else None


def _win32_apply_mingw_toolchain(extra):
    """
    Force MinGW Makefiles + gcc/g++ so CMake does not default to NMake/MSVC.
    Uses W64DEVKIT if set; else gcc on PATH; else %USERPROFILE%\\Downloads\\w64devkit.
    """
    if sys.platform != "win32":
        return

    def _prepend_path(bin_dir):
        path = os.environ.get("PATH", "")
        extra["PATH"] = f"{bin_dir}{os.pathsep}{path}"

    def _set_gcc_from_bin(bin_dir):
        gcc = os.path.join(bin_dir, "gcc.exe")
        gxx = os.path.join(bin_dir, "g++.exe")
        if not (os.path.isfile(gcc) and os.path.isfile(gxx)):
            return False
        _prepend_path(bin_dir)
        if not os.environ.get("CMAKE_GENERATOR"):
            extra.setdefault("CMAKE_GENERATOR", "MinGW Makefiles")
        if not os.environ.get("CC"):
            extra["CC"] = gcc
        if not os.environ.get("CXX"):
            extra["CXX"] = gxx
        return True

    env_root = os.environ.get("W64DEVKIT", "").strip()
    if env_root:
        if not os.path.isdir(env_root):
            raise RuntimeError(f"W64DEVKIT is set but not a directory: {env_root}")
        if not _set_gcc_from_bin(os.path.join(env_root, "bin")):
            raise RuntimeError(
                f"W64DEVKIT bin must contain gcc.exe and g++.exe: {os.path.join(env_root, 'bin')}"
            )
        return

    gcc_path, gxx_path = shutil.which("gcc"), shutil.which("g++")
    if gcc_path and gxx_path:
        if not os.environ.get("CMAKE_GENERATOR"):
            extra.setdefault("CMAKE_GENERATOR", "MinGW Makefiles")
        if not os.environ.get("CC"):
            extra.setdefault("CC", gcc_path)
        if not os.environ.get("CXX"):
            extra.setdefault("CXX", gxx_path)
        return

    fb = _w64devkit_fallback_root()
    if fb and _set_gcc_from_bin(os.path.join(fb, "bin")):
        return

    raise RuntimeError(
        "No MinGW gcc/g++ found. Either add w64devkit\\bin to PATH, set W64DEVKIT to the kit root "
        f"(e.g. C:/Users/YOU/Downloads/w64devkit), or install w64devkit under "
        f"{os.path.join(os.path.expanduser('~'), *_W64DEVKIT_FALLBACK_PARTS)}."
    )


def _source_build_env(*cmake_flag_groups):
    """Env for pip source builds: CMAKE_ARGS plus MinGW (w64devkit or PATH) to avoid NMake."""
    extra = {}
    cenv = _cmake_env(*cmake_flag_groups)
    if cenv:
        extra.update(cenv)
    _win32_apply_mingw_toolchain(extra)
    return extra or None


def _run(cmd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if sys.platform == "win32":
        cflags = env.get("CFLAGS", "")
        env["CFLAGS"] = f"{cflags} -D_WIN32_WINNT=0x0A00".strip()
        cxxflags = env.get("CXXFLAGS", "")
        env["CXXFLAGS"] = f"{cxxflags} -D_WIN32_WINNT=0x0A00".strip()
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def install_base():
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    _run([sys.executable, "-m", "pip", "install", "-r", req_file])


def _require_prebuilt_cu126_wheel_compatible():
    """
    jamepeng-cu126 installs a single fixed wheel (cp311, win_amd64).
    Fail fast with actionable errors instead of a cryptic pip resolution message.
    """
    if sys.platform != "win32":
        raise RuntimeError(
            "Profile 'jamepeng-cu126' only provides a prebuilt Windows wheel. "
            "On Linux/macOS use: --profile jamepeng-nvidia (CUDA, source build) or --profile cpu."
        )
    if sys.maxsize <= 2**32:
        raise RuntimeError(
            "Profile 'jamepeng-cu126' requires 64-bit Python (win_amd64). "
            "This interpreter appears to be 32-bit."
        )
    if (sys.version_info.major, sys.version_info.minor) != (3, 11):
        raise RuntimeError(
            "The bundled CUDA 12.6 wheel is built for Python 3.11 only (cp311-cp311). "
            f"You are on Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Fix: recreate the venv with 3.11 (e.g. py -3.11 -m venv .venv311), then rerun. "
            "Alternative: python install_runtime.py --profile jamepeng-nvidia "
            "(source build with CUDA; requires MSVC/CMake/CUDA per project docs)."
        )


def install_llama(profile, force=False):
    profile = str(profile).strip().lower()
    if profile in {"jamespeng-amd", "jamepeng-amd"}:
        profile = "jamepeng-amd"
    if profile == "cu126":
        profile = "jamepeng-cu126"

    # Smart check: skip if correct version is already installed
    if not force:
        try:
            from importlib import metadata
            v = metadata.version("llama-cpp-python")
            if profile == "jamepeng-cu126" and v == "0.3.33+cu126.basic":
                print(f"Profile '{profile}' matches installed version {v}. Skipping.")
                return
            if profile == "cpu" and v == "0.3.33" and "+" not in v:
                print(f"Profile '{profile}' matches installed version {v}. Skipping.")
                return
        except:
            pass
    if profile == "cu126":
        profile = "jamepeng-cu126"

    if profile == "jamepeng-cu126":
        _require_prebuilt_cu126_wheel_compatible()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                JAMEPENG_CU126_WHEEL,
            ]
        )
        return

    if profile == "cpu":
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"llama-cpp-python=={LLAMA_CPP_VERSION}",
                "--only-binary=:all:",
                "--upgrade",
                "--extra-index-url",
                CPU_WHEEL_INDEX,
            ]
        )
        return

    if profile in {"jamepeng", "jamepeng-cpu"}:
        _require_supported_python_for_source_llama()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                LLAMA_CPP_FORK_SPEC,
            ],
            extra_env=_source_build_env(_win32_openmp_link_cmake_args()),
        )
        return

    if profile == "jamepeng-nvidia":
        _require_supported_python_for_source_llama()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                LLAMA_CPP_FORK_SPEC,
            ],
            extra_env=_source_build_env("-DGGML_CUDA=on", _win32_openmp_link_cmake_args()),
        )
        return

    if profile == "jamepeng-amd":
        _require_supported_python_for_source_llama()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                LLAMA_CPP_FORK_SPEC,
            ],
            extra_env=_source_build_env("-DGGML_VULKAN=on", _win32_openmp_link_cmake_args()),
        )
        return

    if profile == "nvidia":
        raise RuntimeError(
            "The 'nvidia' profile requires a local C++ compiler for a source build. "
            "To use a prebuilt CUDA wheel instead, use --profile jamepeng-cu126."
        )

    if profile == "amd":
        raise RuntimeError(
            "The 'amd' profile requires a local C++ compiler for a source build. "
            "For maximum compatibility with prebuilt wheels, use --profile cpu."
        )

    raise ValueError(f"Unsupported profile: {profile}")


def main():
    parser = argparse.ArgumentParser(
        description="Install FocusEngine runtime dependencies with CPU/GPU profile."
    )
    parser.add_argument(
        "--profile",
        choices=[
            "cpu",
            "nvidia",
            "amd",
            "jamepeng",
            "jamepeng-cpu",
            "jamepeng-nvidia",
            "jamepeng-amd",
            "jamepeng-cu126",
            "cu126",
            "jamespeng",
            "jamespeng-cpu",
            "jamespeng-nvidia",
            "jamespeng-amd",
        ],
        default="jamepeng-cu126",
        help=(
            "llama-cpp-python profile. "
            "jamepeng-cu126 = prebuilt CUDA 12.6 wheel (Windows x64 + Python 3.11 only)."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force reinstall of llama-cpp-python.")
    args = parser.parse_args()

    print(f"Installing FocusEngine dependencies (profile={args.profile})")
    install_base()
    install_llama(args.profile, force=args.force)
    print("Dependency install complete.")


if __name__ == "__main__":
    main()
