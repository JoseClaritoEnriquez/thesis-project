import os
import sys
import subprocess
import shutil

def build():
    """
    Automates the PyInstaller build process for Focus Engine.
    Ensures that OpenVINO and its dependencies are correctly bundled.
    """
    # 1. Base paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spec_file = os.path.join(script_dir, "FocusEngine.spec")
    dist_dir = os.path.join(script_dir, "dist")
    build_dir = os.path.join(script_dir, "build")

    print("--- Focus Engine: Automated Build Pipeline ---")
    
    # 2. Cleanup previous builds
    if os.path.exists(dist_dir):
        print(f"Cleaning up {dist_dir}...")
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        print(f"Cleaning up {build_dir}...")
        shutil.rmtree(build_dir)

    # 3. Verify Spec file exists
    if not os.path.exists(spec_file):
        print(f"Error: Spec file not found at {spec_file}")
        sys.exit(1)

    # 4. Run PyInstaller
    print(f"Running PyInstaller for {spec_file}...")
    try:
        # We use the spec file because it contains the complex collect_all hooks for OpenVINO
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller", 
            spec_file, 
            "--noconfirm",
            "--clean"
        ])
        print("\nBuild Successful!")
        print(f"Executable folder can be found in: {os.path.join(dist_dir, 'FocusEngine')}")
    except subprocess.CalledProcessError as e:
        print(f"\nBuild Failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
